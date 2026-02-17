import pandas as pd

df = pd.read_csv("Financial_inclusion_dataset.csv")

print("Initial shape:", df.shape)

# حذف uniqueid لأنه غير مفيد
df = df.drop(columns=["uniqueid"])

# حذف التكرارات
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# تحويل target إلى 0 و 1
df["bank_account"] = df["bank_account"].map({"Yes": 1, "No": 0})

print(df["bank_account"].value_counts())

df.to_csv("cleaned_data.csv", index=False)

print("Cleaned data saved.")
