import argparse

import numpy as np
from ase.db import connect
from matminer.featurizers.site import AverageBondAngle
from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.site.bonding import AverageBondLength
from ase import Atoms
from tqdm import tqdm
from kan import *
import sys
from pymatgen.core import Structure
from ase.build import bulk
from ase import Atom, Atoms
from dscribe.descriptors import SOAP
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
from pymatgen.io.ase import AseAtomsAdaptor

parser = argparse.ArgumentParser(description='KAN for mossbauer')
parser.add_argument('--predict_item', choices=['mm', 'efg', 'rto', 'eta', 'hff'], default='eta',
                    help="可选预测值:'mm', 'efg', 'rto', 'eta', 'hff'")
parser.add_argument('--steps', default=100, type=int, help='number of steps in KAN model fit') # 100
parser.add_argument('--k', default=3, type=int, help='number of k in KAN model init')
parser.add_argument('--grids', type=str, default='3,10,20,50,100', help='list of integers')  # '3,10,20,50,100'
parser.add_argument('--model_width', type=str, default='4,4,2,1',
                    help='Comma-separated list of model widths')
args = parser.parse_args()


def extract_descriptor(rows):
    featurizers_site, targets = [], []  # 存储 描述符 和 目标值
    for row in tqdm(rows, desc="获取结构描述信息", leave=True):
        atoms_Au_Fe = row.toatoms()  # 转为原子对象
        atoms_all_Fe = Atoms()
        atoms_all_Fe.set_cell(atoms_Au_Fe.get_cell())  # 设置晶胞参数
        atoms_all_Fe.set_pbc(atoms_Au_Fe.get_pbc())  # 设置周期性边界条件
        Au_idx_lst = []  # 存储转换前的Au的索引
        # 将Au转化为Fe，并记录Au的位置
        for idx, at in enumerate(atoms_Au_Fe):
            if at.symbol == 'Fe':
                atoms_all_Fe.append(Atom(at.symbol, at.position))
            elif at.symbol == 'Au':
                atoms_all_Fe.append(Atom('Fe', at.position))  # 将金原子替换为铁原子并添加到新的原子对象中
                Au_idx_lst.append(idx)  # 记录金原子的索引
            else:
                atoms_all_Fe.append(Atom(at.symbol, at.position))

        # 将 ASE 的 Atoms 对象转换为 Pymatgen 的 Structure 对象
        structure = AseAtomsAdaptor.get_structure(atoms_all_Fe)

        cutoff = 3
        vnn = VoronoiNN(cutoff=cutoff)
        all_descriptor = []
        Fe_count = []  # Fe原子数量
        S_count = []  # S原子数量
        avg_BondAngle = []  # 平均键角
        avg_BondLength = []  # 平均键长
        # 键长 键角
        BondAngle = AverageBondAngle(vnn)
        BondLength = AverageBondLength(vnn)
        try:
            # 计算每个 Au 原子的描述符
            for au_idx in Au_idx_lst:
                # 统计 Fe 和 S 原子的数量
                fe_count = 0
                s_count = 0
                neighbors = vnn.get_nn_info(structure, au_idx)
                for neighbor in neighbors:
                    element = neighbor['site'].species_string
                    if element == "Fe":
                        fe_count += 1
                    elif element == "S":
                        s_count += 1
                Fe_count.append(fe_count)
                S_count.append(s_count)
                descriptor_angle = BondAngle.featurize(structure, au_idx)
                descriptor_length = BondLength.featurize(structure, au_idx)
                avg_BondAngle.append(descriptor_angle)
                avg_BondLength.append(descriptor_length)

            all_descriptor.extend([np.mean(Fe_count)])
            all_descriptor.extend([np.mean(S_count)])
            all_descriptor.extend(np.mean(avg_BondAngle, axis=0))
            all_descriptor.extend(np.mean(avg_BondLength, axis=0))
            featurizers_site.append(all_descriptor)

            if args.predict_item == 'rto':
                targets.append([row.data[predict_item] / 10000])
            elif args.predict_item == 'hff':
                targets.append([row.data[predict_item] / 10])
            else:
                targets.append([row.data[predict_item]])
        except ValueError as e:
            print(f"cutoff = {cutoff} 时，{atoms_Au_Fe}没有相邻元素")
            continue
    return featurizers_site, targets


def formula_mae(formula_, X, y):
    batch = X.shape[0]
    total_error = 0
    for i in range(batch):
        # 将特征值代入公式，计算输出
        prediction = np.array(
            formula_.subs('x_1', X[i, 0])
            .subs('x_2', X[i, 1])
            .subs('x_3', X[i, 2])
            .subs('x_4', X[i, 3])
        ).astype(np.float64)
        if args.predict_item == 'rto':
            pre = prediction*10000
            la = y[i]*10000
        elif args.predict_item == 'hff':
            pre = prediction * 10
            la = y[i] * 10
        else:
            pre = prediction
            la = y[i]
        total_error += np.abs(pre - la)
    # 计算平均绝对误差
    mean_absolute_error = total_error / batch
    return mean_absolute_error


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    # seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    # 数据集设置
    rcut = 6.0
    nmax = 8
    lmax = 6
    predict_item = args.predict_item

    # 训练数据
    db_train = 'mossbauer_train.db'
    db1 = connect(db_train)
    rows1 = list(db1.select())
    random.shuffle(rows1)
    train_lst, tgt_lst = extract_descriptor(rows1)

    # 测试数据
    db_test = 'mossbauer_test.db'
    db2 = connect(db_test)
    rows2 = list(db2.select())
    test_lst, test_tgt_lst = extract_descriptor(rows2)

    train_input = torch.from_numpy(np.array(train_lst)).double().to('cpu')
    train_label = torch.from_numpy(np.array(tgt_lst)).double().to('cpu')
    test_input = torch.from_numpy(np.array(test_lst)).double().to('cpu')
    test_label = torch.from_numpy(np.array(test_tgt_lst)).double().to('cpu')

    # 创建 dataset 字典
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    # 输入、输出维度
    input_dim = len(train_input[0])
    output_dim = len(train_label[0])

    # 迭代
    grids = np.fromstring(args.grids, dtype=int, sep=',')
    train_losses = []
    test_losses = []
    steps = args.steps
    k = args.k  # 样条函数的阶数

    model_width = list(map(int, args.model_width.split(',')))
    # model = KAN(width=model_width, grid=3, k=k, seed=seed, auto_save=True, base_fun='identity')  # 'identity' 'silu' affine_trainable=True

    def train_mae():
        predictions = model(dataset['train_input'])
        labels = dataset['train_label']
        mae = torch.mean(torch.abs(predictions - labels))
        return mae


    def test_mae():
        predictions = model(dataset['test_input'])
        labels = dataset['test_label']
        mae = torch.mean(torch.abs(predictions - labels))
        return mae

    for i in range(grids.shape[0]):
        if i == 0:
            print("train with grid = ", grids[i])
            model = KAN(width=model_width, grid=grids[i], k=k, seed=seed, auto_save=True, base_fun='identity')
        else:
            print("change grid = ", grids[i])
            model = model.refine(grids[i])

        # model = model.speed()
        results = model.fit(dataset, opt="LBFGS", steps=steps, lamb=1e-3, metrics=(train_mae, test_mae))
        train_losses += results['train_loss']
        test_losses += results['test_loss']

    # 公式
    # lib = ['x', 'x^2', 'x^3', 'x^4', 'x^5', '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 'sqrt', 'x^0.5', 'x^1.5','1/sqrt(x)','1/x^0.5', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn', 'arcsin', 'arccos', 'arctan', 'arctanh','0','gaussian']
    lib = ['x', 'x^2', 'exp', 'log', 'tanh', 'sin', 'abs']
    model.auto_symbolic(lib=lib)

    print(model.suggest_symbolic(0, 0, 0))
    # 符号公式
    formula = model.symbolic_formula()[0][0]
    print(formula)

    # Calculate mae on training and test sets
    print('Train mae of the formula:',
          formula_mae(formula, dataset['train_input'].numpy(), dataset['train_label'].numpy()))
    print('Test mae of the formula:',
          formula_mae(formula, dataset['test_input'].numpy(), dataset['test_label'].numpy()))
