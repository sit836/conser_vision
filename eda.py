import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

from constants import IN_PATH

train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'))
train_features = pd.read_csv(os.path.join(IN_PATH, 'train_features.csv'))
test_features = pd.read_csv(os.path.join(IN_PATH, 'test_features.csv'))

print(train_labels.isnull().sum())
# print(train_labels.iloc[:, 1:].sum(axis=1).max())

# balance
print(train_labels.iloc[:, 1:].sum(axis=0) / len(train_labels))
# antelope_duiker     0.150049
# bird                0.099527
# blank               0.134219
# civet_genet         0.146955
# hog                 0.059316
# leopard             0.136705
# monkey_prosimian    0.151140
# rodent              0.122089

category = 'leopard'
is_selected = train_labels[category] == 1
ids_to_move = train_labels.loc[is_selected, 'id']
file_path = 'D:/py_projects/Conser_vision_Practice_Area/data/train_features/'
save_path = f'D:/py_projects/Conser_vision_Practice_Area/{category}/'

for f in tqdm(os.listdir(file_path)):
    if f[:-4] in ids_to_move.values:
        print(f)
        shutil.copyfile(os.path.join(file_path, f), os.path.join(save_path, f))
