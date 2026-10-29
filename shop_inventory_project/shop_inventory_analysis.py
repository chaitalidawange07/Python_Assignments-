import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save charts reliably on computers without a GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

print("=" * 50)
print("SHOP INVENTORY DATA ANALYSIS & DASHBOARD")
print("=" * 50)

file_name = "shop_data.csv"
df = pd.read_csv(file_name)

print("\n[INFO] Initializing Data Cleaning...")
print(f"Original dataset shape: {df.shape} (Checking for Null Values)")
df_clean = df.dropna().copy()
print(f"Cleaned dataset shape:  {df_clean.shape} (Null Values Removed)")

if len(df_clean) < 10:
    raise ValueError("At least 10 complete rows are recommended for this demonstration.")
if df_clean["Out_of_Stock"].nunique() < 2:
    raise ValueError("Out_of_Stock must contain both 0 and 1 for logistic regression.")

print("\n" + "-" * 50)
print("MODEL 1: LINEAR REGRESSION")
print("-" * 50)
X_lin = df_clean[["Price", "Daily_Visitors"]]
y_lin = df_clean["Units_Sold"]
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)
lin_predictions = lin_model.predict(X_test_lin)
mse = mean_squared_error(y_test_lin, lin_predictions)
print("[SUCCESS] Linear Regression model trained.")
print(f"[RESULT] Mean Squared Error: {mse:.2f}")

print("\n" + "-" * 50)
print("MODEL 2: LOGISTIC REGRESSION")
print("-" * 50)
X_log = df_clean[["Units_Sold"]]
y_log = df_clean["Out_of_Stock"]
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_log, y_train_log)
log_predictions = log_model.predict(X_test_log)
accuracy = accuracy_score(y_test_log, log_predictions)
print("[SUCCESS] Logistic Regression model trained.")
print(f"[RESULT] Classification Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

# Four-chart dashboard
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comprehensive Shop Data Analysis", fontsize=16, fontweight="bold")

axs[0, 0].scatter(df_clean["Price"], df_clean["Units_Sold"], alpha=0.7)
axs[0, 0].set_title("Scatter Plot: Price vs. Units Sold")
axs[0, 0].set_xlabel("Price")
axs[0, 0].set_ylabel("Units Sold")
axs[0, 0].grid(True, linestyle="--", alpha=0.5)

axs[0, 1].hist(df_clean["Units_Sold"], bins=15, edgecolor="black", alpha=0.75)
axs[0, 1].set_title("Histogram: Distribution of Units Sold")
axs[0, 1].set_xlabel("Units Sold")
axs[0, 1].set_ylabel("Frequency")
axs[0, 1].grid(True, linestyle="--", alpha=0.5)

subset = df_clean.head(30)
axs[1, 0].plot(subset["Product_ID"], subset["Units_Sold"], marker="o", linewidth=1.5)
axs[1, 0].set_title("Line Graph: Units Sold for First 30 Products")
axs[1, 0].set_xlabel("Product ID")
axs[1, 0].set_ylabel("Units Sold")
axs[1, 0].grid(True, linestyle="--", alpha=0.5)

columns = ["Price", "Daily_Visitors", "Units_Sold", "Out_of_Stock"]
corr = df_clean[columns].corr()
cax = axs[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax, ax=axs[1, 1])
axs[1, 1].set_xticks(np.arange(len(columns)))
axs[1, 1].set_yticks(np.arange(len(columns)))
axs[1, 1].set_xticklabels(columns, rotation=45, ha="right")
axs[1, 1].set_yticklabels(columns)
axs[1, 1].set_title("Heat Map: Feature Correlation")
for i in range(len(columns)):
    for j in range(len(columns)):
        axs[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                       color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")

fig.tight_layout(h_pad=3, w_pad=2)
fig.subplots_adjust(top=0.90)
fig.savefig("dashboard_output.png", dpi=160, bbox_inches="tight")
print("[OUTPUT] Dashboard saved as dashboard_output.png")
print("[INFO] Run completed successfully.")
