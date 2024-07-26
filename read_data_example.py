import sys

from ase.db import connect
from ase.visualize import view

db_name1 = 'result/mossbauer.db'
db_name2 = 'mossbauer.db'
db_name3 = 'mossbauer_train.db'
db_name4 = 'mossbauer_test.db'
db = connect(db_name2)
print(db.count())
# for row in db.select():
#
#     print('row.id: ', row.id)
#     print('REGRESSION INPUT: ')
#     print('cell of crystall: ', row.toatoms().get_cell())
#     print('enviroment atoms: ')
#     for at in row.toatoms():
#         if at.symbol != 'Au':
#             print(at.position, at.symbol)
#     print('Contributed Fe atoms: ')
#     for at in row.toatoms():
#         if at.symbol == 'Au':
#             print(at.position, 'Fe')
#     print('REGRESSION OUTPUT: ')
#     print('data need to regress: ', row.data)
#     print('-' * 100)
#
#     # if row.id == 1:
#     #     view(row.toatoms())

# db = connect(db_name4)
# print(db.count())
# count2 = 0
# for row in db.select():
#     count2+=1
# print(count2)