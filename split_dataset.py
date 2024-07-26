import os
import sys

import ase
from ase.db import connect
import random
from tqdm import tqdm

# TODO: use re to replace scf parser
import extracted_data as extct_mh

root_folder = './dataset/MES_data'
max_iron_cnt = 10000

db_name1 = 'mossbauer_train.db'
db_name2 = 'mossbauer_test.db'

os.system('rm -rf ' + db_name1)
os.system('rm -rf ' + db_name2)
db_train = connect(db_name1)
db_test = connect(db_name2)


# parse workflow:
# 1. traverse every folder under root
# 2. check 2 files, same name but diff extension (scf and struct)
# 3. read struct file
# 3.1 copy struct file, replace some Fe to Au
# 3.2 read new struct file
# 4. parse scf find 5 value. (MaHuang contribute the code)
# 5. write into db

def parse_struc_file(file_name):
    # print('stuc func:', file_name)
    fe_au_atoms = []
    iron_idx = extct_mh.get_Fe_atoms(file_name)
    with open(file_name, 'r') as orig_fd:
        orig_str = orig_fd.readlines()  # very small file

    for line_idx in range(len(orig_str)):
        if 'Fe' in orig_str[line_idx]:
            orig_str[line_idx] = orig_str[line_idx].replace('Fe', 'Au', 1)
            tmp_file_name = 'tmp.struct'
            with open(tmp_file_name, 'w') as tmp_fd:
                tmp_fd.writelines(orig_str)
            at = ase.io.read(tmp_file_name)
            fe_au_atoms.append(at)
            orig_str[line_idx] = orig_str[line_idx].replace('Au', 'Fe', 1)

    return iron_idx, fe_au_atoms


def parse_scf_file(file_name, iron_idx):
    # print('scf func:', file_name, iron_idx)
    mm = extct_mh.get_MM(file_name, iron_idx)
    hff = extct_mh.get_HFF(file_name, iron_idx)
    eta = extct_mh.get_ETA(file_name, iron_idx)
    efg = extct_mh.get_EFG(file_name, iron_idx)
    rto = extct_mh.get_RTO(file_name, iron_idx)
    ret = []
    for i in range(len(iron_idx)):
        ret.append((mm[i], hff[i], eta[i], efg[i], rto[i]))
    return ret


def process_subdirs(subdirs, db):
    for roots in tqdm(subdirs, desc="Processing directories"):
        # print('Folder: ', roots)
        for root, dirs, files in os.walk(roots):
            if len(files) != 2:
                # print('file count != 2, pass')
                # print('-' * 100)
                continue

            struct_name, scf_name = '', ''
            for f in files:
                file_name = os.path.join(root, f)
                # print(file_name)
                if f.endswith('.struct') and struct_name == '':
                    struct_name = file_name
                elif f.endswith('.scf') and scf_name == '':
                    scf_name = file_name
                else:
                    # print('file format error.')
                    # print('-' * 100)
                    break
            else:
                iron_idx, at_lst = parse_struc_file(struct_name)
                props_lst = parse_scf_file(scf_name, iron_idx)

                # print(at_lst, props_lst)
                if len(at_lst) != len(props_lst):
                    # print('struct file and scf file is not a pair? diff length...')
                    # print('-' * 100)
                    continue
                else:
                    for i in range(len(at_lst)):
                        if '' not in props_lst[i]:
                            db.write(at_lst[i], data={'mm': float(props_lst[i][0]),
                                                      'hff': float(props_lst[i][1]),
                                                      'eta': float(props_lst[i][2]),
                                                      'efg': float(props_lst[i][3]),
                                                      'rto': float(props_lst[i][4])
                                                      })


def main():
    random.seed(42)
    print('parse start: ')
    # Step 1: Get all direct subdirectories in root_folder
    subdirs = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if
               os.path.isdir(os.path.join(root_folder, d))]
    # subdirs = [os.path.join(root, d) for root, dirs, files in os.walk(root_folder) for d in dirs]

    # Step 2: Randomly select 75% of the subdirectories
    selected_subdirs1 = random.sample(subdirs, k=int(len(subdirs) * 0.8))
    selected_subdirs2 = [d for d in subdirs if d not in selected_subdirs1]

    # Process the selected subdirectories
    print('Processing training directories...')
    process_subdirs(selected_subdirs1, db_train)

    print('Processing testing directories...')
    process_subdirs(selected_subdirs2, db_test)


if __name__ == '__main__':
    main()
