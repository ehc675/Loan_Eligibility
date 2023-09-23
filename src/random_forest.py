import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

df = pd.read_csv("../dataset/train.csv")
print(df)