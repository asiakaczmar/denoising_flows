from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import accuracy_score

from utils import get_parser_model_flow
from utils.classification import load_dataframes, Y_COLUMN, predict, DEVICE

RANDOM_STATE = 42

results = []

args = get_parser_model_flow().parse_args()
path = args.model_path

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE)
df_train, df_test = load_dataframes(path)

print(df_train.shape, df_train.columns)
print(df_test.shape, df_train.columns)

x_train, y_train = df_train.drop(Y_COLUMN, axis=1), df_train[Y_COLUMN]
x_test, y_test = df_test.drop(Y_COLUMN, axis=1), df_test[Y_COLUMN]

x_train = torch.tensor(x_train.values, dtype=torch.float)
x_test = torch.tensor(x_test.values, dtype=torch.float)

num_classes = 10

y_train_hat = predict(flow, x_train, num_classes)
y_test_hat = predict(flow, x_test, num_classes)

results.append(('flow', 'train', accuracy_score(y_train, y_train_hat)))
results.append(('flow', 'test', accuracy_score(y_test, y_test_hat)))

print('Saving results.')
results = pd.DataFrame(results)
results.to_csv(args.flow_path / Path('classification-results-flow.csv'), index=False)
