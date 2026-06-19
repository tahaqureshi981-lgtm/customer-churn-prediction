import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/telco_churn.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 3 rows:\n", df.head(3))

# ── Basic Info ────────────────────────────────────────────────────────────────
print("\n── Dtypes & Nulls ──")
print(df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts())
print("Churn rate:", round(df["Churn"].value_counts(normalize=True)["Yes"] * 100, 2), "%")

# Fix TotalCharges — it's loaded as string due to spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
null_after_fix = df["TotalCharges"].isnull().sum()
print(f"\nTotalCharges nulls after fix: {null_after_fix} (these are new customers — tenure=0)")

# ── Plot 1: Churn Distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#4C9BE8", "#E8614C"], ax=ax, edgecolor="white")
ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
ax.set_xticklabels(["No", "Yes"], rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height() + 30),
                ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/churn_distribution.png", dpi=150)
plt.close()
print("\n✅ Saved: churn_distribution.png")

# ── Plot 2: Churn by Contract Type ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ct = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
ct.columns = ["Contract", "ChurnRate"]
sns.barplot(data=ct, x="Contract", y="ChurnRate", palette="Blues_d", ax=ax)
ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/churn_by_contract.png", dpi=150)
plt.close()
print("✅ Saved: churn_by_contract.png")

# ── Plot 3: Tenure vs Churn ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color in [("Yes", "#E8614C"), ("No", "#4C9BE8")]:
    df[df["Churn"] == label]["tenure"].plot(kind="kde", ax=ax, label=f"Churn={label}", color=color, linewidth=2)
ax.set_title("Tenure Distribution by Churn", fontsize=14, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/tenure_vs_churn.png", dpi=150)
plt.close()
print("✅ Saved: tenure_vs_churn.png")

# ── Plot 4: Monthly Charges vs Churn ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, color in [("Yes", "#E8614C"), ("No", "#4C9BE8")]:
    df[df["Churn"] == label]["MonthlyCharges"].plot(kind="kde", ax=ax, label=f"Churn={label}", color=color, linewidth=2)
ax.set_title("Monthly Charges Distribution by Churn", fontsize=14, fontweight="bold")
ax.set_xlabel("Monthly Charges ($)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/monthly_charges_vs_churn.png", dpi=150)
plt.close()
print("✅ Saved: monthly_charges_vs_churn.png")

# ── Plot 5: Correlation Heatmap (numeric only) ────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Saved: correlation_heatmap.png")

# ── Key Insights Summary ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("📊 KEY EDA INSIGHTS")
print("="*50)
print(f"  Total customers     : {len(df):,}")
print(f"  Churn rate          : {(df['Churn']=='Yes').mean()*100:.1f}%  ← class imbalance!")
print(f"  Avg tenure (churned): {df[df['Churn']=='Yes']['tenure'].mean():.1f} months")
print(f"  Avg tenure (stayed) : {df[df['Churn']=='No']['tenure'].mean():.1f} months")
print(f"  Avg monthly charges (churned): ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}")
print(f"  Avg monthly charges (stayed) : ${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}")
m2m_churn = df[df["Contract"]=="Month-to-month"]["Churn"].value_counts(normalize=True)["Yes"]*100
print(f"  Month-to-month churn rate    : {m2m_churn:.1f}%  ← highest risk segment")
print("="*50)
print("\n✅ EDA complete. All charts saved to reports/")
