import pandas as pd

df = pd.read_csv("cleaned_data.csv")

cols_to_clip = ["age_of_respondent", "household_size"]

for col in cols_to_clip:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df[col] = df[col].clip(low, high)

df.to_csv("final_data.csv", index=False)

print("Outliers handled and saved to final_data.csv")
print(df[cols_to_clip].describe())
