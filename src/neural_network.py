import pandas as pd
import torch

def read_data(train_location, train_):
    df = pd.read_csv(data_location)
    x_train = torch.tensor(df.iloc[0:split_index, 1:].values, dtype=torch.float32)
    y_train = torch.tensor(df.iloc[0:split_index, 0].values, dtype=torch.float32)
    #print(x)
    x_test = torch.tensor(df.iloc[split_index:, 1:].values, dtype=torch.float32)
    y_test = torch.tensor(df.iloc[split_index:, 0].values, dtype=torch.float32)
    print(x_train.shape)
    print(x_test.shape)

def model():
    
