import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../data.csv")

print(df.head())
print(df.columns.tolist())
print(df.describe())

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns