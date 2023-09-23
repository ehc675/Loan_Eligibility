import pandas as pd
import os

def preProcess(data_location) -> None:
    df = pd.read_csv(data_location)
    #Shuffle input
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    split_index = int(df.shape[0]*0.8)
    df_train = df[0:split_index]
    df_test = df[split_index:]
    df_train.to_csv(os.path("/dataset/cleaned_train.csv"))
    df_test.to_csv(os.path("/dataset/cleaned_test"))