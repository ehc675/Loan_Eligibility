import pandas as pd
import os

def preProcess(data_location) -> None:
    print(os.chdir("./../dataset"))
    print(os.listdir())
    df = pd.read_csv(data_location)
    #Shuffle input
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    split_index = int(df.shape[0]*0.8)
    df_train = df[0:split_index]
    df_test = df[split_index:]
    df_train.to_csv("cleaned_train.csv")
    df_test.to_csv("cleaned_test.csv")

# preProcess("clean_1e5.csv")

def fillNaN(data_location, target_location):
    df = pd.read_csv(data_location)
    column_means = df.mean()
    df.fillna(column_means, inplace = True)
    df.to_csv(target_location, index = False)

# fillNaN("../dataset/cleaned_train.csv", "../dataset/filled_train.csv")
# fillNaN("../dataset/cleaned_test.csv", "../dataset/filled_test.csv")