"""Cyber Crime Data Analysis and Visualization in India Using Python
Authors: Vishal Shinde (SECC157), Vishal Yadav (SECC168)
Run: python cyber_crime_analysis.py   (needs cyber_crime_india.csv in same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

# ---------- 1. Load and inspect ----------
df = pd.read_csv("cyber_crime_india.csv")
print(df.head()); print(df.shape); df.info()
print("Missing before cleaning:\n", df.isnull().sum())

# ---------- 2. Clean ----------
df.columns = df.columns.str.strip()
df = df[["State", "Year", "Cases", "Charge_Sheeted", "Arrested"]]
df["State"] = df["State"].astype(str).str.strip().str.title()
df.loc[df["State"].isin(["Nan", "None", ""]), "State"] = np.nan

num_cols = ["Cases", "Charge_Sheeted", "Arrested"]
for c in num_cols + ["Year"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in num_cols:
    df.loc[df[c] < 0, c] = np.nan

df = df.dropna(subset=["State", "Year"])
dups = df.duplicated().sum()
df = df.drop_duplicates()
print("Duplicates removed:", dups)

for c in num_cols:
    df[c] = df.groupby("State")[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())
df["Year"] = df["Year"].astype(int)
print("Missing after cleaning:\n", df.isnull().sum())
df.to_csv("outputs/cleaned_data.csv", index=False)

# ---------- 3. NumPy analysis ----------
cases = df["Cases"].to_numpy()
print("Mean  :", np.mean(cases))
print("Median:", np.median(cases))
print("Min   :", np.min(cases))
print("Max   :", np.max(cases))
print("Total :", np.sum(cases))
print("Std   :", np.std(cases))
print("P75   :", np.percentile(cases, 75))

yearly = df.groupby("Year")["Cases"].sum()
growth = np.diff(yearly.to_numpy()) / yearly.to_numpy()[:-1] * 100
print("\nYear-wise totals:\n", yearly)
print("Growth %:", dict(zip(yearly.index[1:], np.round(growth, 2))))

state_total = df.groupby("State")["Cases"].sum().sort_values(ascending=False)
share = state_total / state_total.sum() * 100
print("\nTop 3 states:\n", pd.DataFrame({"Total": state_total, "Share %": share.round(2)}).head(3))

corr = np.corrcoef(df["Cases"], df["Arrested"])[0, 1]
print("Correlation (Cases, Arrested):", round(corr, 3))

# ---------- 4. Visualisation ----------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.barplot(x=state_total.head(10).values, y=state_total.head(10).index, color="#2b6cb0")
plt.title("Top 10 States by Total Cyber Crime Cases")
plt.xlabel("Total cases"); plt.ylabel("State"); plt.tight_layout()
plt.savefig("outputs/1_top_states.png", dpi=150); plt.close()

plt.figure(figsize=(9, 5))
plt.plot(yearly.index, yearly.values, marker="o", color="#c53030")
plt.title("Year-wise Total Cyber Crime Cases in India")
plt.xlabel("Year"); plt.ylabel("Total cases"); plt.xticks(yearly.index); plt.tight_layout()
plt.savefig("outputs/2_yearly_trend.png", dpi=150); plt.close()

plt.figure(figsize=(9, 5))
plt.hist(cases, bins=20, color="#2f855a", edgecolor="white")
plt.title("Distribution of Cases (State-Year Records)")
plt.xlabel("Cases"); plt.ylabel("Frequency"); plt.tight_layout()
plt.savefig("outputs/3_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["Cases"], df["Arrested"], alpha=0.6, color="#6b46c1")
m, b = np.polyfit(df["Cases"], df["Arrested"], 1)
xs = np.linspace(df["Cases"].min(), df["Cases"].max(), 100)
plt.plot(xs, m * xs + b, color="black", label="Trend line")
plt.title("Cases vs Arrests"); plt.xlabel("Cases registered"); plt.ylabel("Persons arrested")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/4_scatter.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df[["Year"] + num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix"); plt.tight_layout()
plt.savefig("outputs/5_heatmap.png", dpi=150); plt.close()
print("All graphs saved in outputs/")
