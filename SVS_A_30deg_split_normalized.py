import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# 1. Load data
# =========================
file_path = "SVS_A_30deg_split_normalized.xlsx"

train = pd.read_excel(file_path, sheet_name="Train_normalized")
test  = pd.read_excel(file_path, sheet_name="Test_normalized")

# =========================
# 2. Define variables
# =========================
X_train = train[["e_Dap_norm", "DeltaT_norm", "U_norm"]].copy()
X_test  = test[["e_Dap_norm", "DeltaT_norm", "U_norm"]].copy()

y_train = train["V_throat"]
y_test  = test["V_throat"]

# =========================
# 3. Add interaction term
# =========================
X_train["interaction"] = X_train["e_Dap_norm"] * X_train["U_norm"]
X_test["interaction"]  = X_test["e_Dap_norm"] * X_test["U_norm"]

# =========================
# 4. Add constant
# =========================
X_train = sm.add_constant(X_train)
X_test  = sm.add_constant(X_test)

# =========================
# 5. Fit model (OLS)
# =========================
model = sm.OLS(y_train, X_train).fit()

print(model.summary())

# =========================
# 6. Predictions
# =========================
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# =========================
# 7. Evaluation metrics
# =========================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

r2_tr, rmse_tr, mae_tr = evaluate(y_train, y_pred_train)
r2_te, rmse_te, mae_te = evaluate(y_test, y_pred_test)

print("\n=== Train Performance ===")
print(f"R2   = {r2_tr:.4f}")
print(f"RMSE = {rmse_tr:.4f}")
print(f"MAE  = {mae_tr:.4f}")

print("\n=== Test Performance ===")
print(f"R2   = {r2_te:.4f}")
print(f"RMSE = {rmse_te:.4f}")
print(f"MAE  = {mae_te:.4f}")
# =========================
# Visualization + Save
# =========================
import matplotlib.pyplot as plt
import os

# مسیر ذخیره (همون مسیر فایل اکسل)
save_path = os.path.dirname(file_path)

# -------- Predicted vs Actual --------
plt.figure()
plt.scatter(y_test, y_pred_test)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)])

plt.xlabel("Actual V_throat")
plt.ylabel("Predicted V_throat")
plt.title("Predicted vs Actual (Test Data)")
plt.grid()

plt.savefig(os.path.join(save_path, "Predicted_vs_Actual.png"), dpi=300)
plt.show()


# -------- Residual Plot --------
residuals = y_test - y_pred_test

plt.figure()
plt.scatter(y_pred_test, residuals)
plt.axhline(0)

plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid()

plt.savefig(os.path.join(save_path, "Residual_Plot.png"), dpi=300)
plt.show()


# -------- Histogram --------
plt.figure()
plt.hist(residuals, bins=10)

plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.savefig(os.path.join(save_path, "Residual_Histogram.png"), dpi=300)
plt.show()
