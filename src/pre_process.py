import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preProcess(data_location) -> None:
    """
    split big dataset into train and test dataset (8:2)
    """
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


def fillNaN(data_location, target_location):
    """
    fill in NaN with average values of columns
    """
    df = pd.read_csv(data_location)
    column_means = df.mean()
    df.fillna(column_means, inplace = True)
    df.to_csv(target_location, index = False)



def categorical():
    """
    turn dwelling and ethnicity data (strings) into numeric labels
    """
    train = pd.read_csv("../dataset/cleaned_train.csv")
    test = pd.read_csv("../dataset/cleaned_test.csv")
    label_encoder = LabelEncoder()

    columns_to_encode = ["derived_dwelling_category", "derived_ethnicity", "applicant_age"]
    for column in columns_to_encode:
        train[column] = label_encoder.fit_transform(train[column])
        test[column] = label_encoder.fit_transform(test[column])
    train = train[train['interest_rate'] != "Exempt"]
    test = test[test['interest_rate'] != "Exempt"]
    train = train.drop(train[train['denial_reason_1'] == 1111].index)
    test = test.drop(test[test['denial_reason_1'] == 1111].index)
    
    train = train.replace("","NaN")
    test = test.replace("","NaN")
    train = train.replace(' ',"NaN")
    test = test.replace(' ',"NaN")
    
    # calculate the mean of each column
    train_mean = train.mean()
    #print(train_mean)
    # fill NaN values with the mean of each column
    train.fillna(train_mean, inplace=True)
    test_mean = test.mean()
    test.fillna(test_mean, inplace=True)
    print(train["interest_rate"])
    train.to_csv("../dataset/labelled_train.csv")
    test.to_csv("../dataset/labelled_test.csv")
    

#def delete_Exempt(df):
#    """
#    delete ther rows with value "Exempt" in one of its column
#    """
#    df_filtered = df[df['interest_rate'] != "Exempt"]
#    return df_filtered



if __name__ == '__main__':
    # preProcess("clean_1e5.csv")
    # fillNaN("../dataset/cleaned_train.csv", "../dataset/filled_train.csv")
    # fillNaN("../dataset/cleaned_test.csv", "../dataset/filled_test.csv")
    fillNaN("../dataset/labelled_train.csv", "../dataset/filled_train.csv")
    fillNaN("../dataset/labelled_test.csv", "../dataset/filled_test.csv")
    categorical()
    #delete_Exempt()