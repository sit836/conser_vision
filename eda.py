import os
import pandas as pd

from constants import IN_PATH

train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'))
print(train_labels)

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
