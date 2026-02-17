import pandas as pd

df = pd.read_csv("Financial_inclusion_dataset.csv")

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
