import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Load data
file_path = "SVS_A_30deg_split_normalized.xlsx"

train = pd.read_excel(file_path, sheet_name="Train_normalized")
test  = pd.read_excel(file_path, sheet_name="Test_normalized")

# Load raw (non-normalized) data to get mean and std
train_raw = pd.read_excel(file_path, sheet_name="Train_raw")

means = train_raw[["e_Dap", "DeltaT", "U"]].mean()
stds  = train_raw[["e_Dap", "DeltaT", "U"]].std()

print("\nMEANS:")
print(means)

print("\nSTDS:")
print(stds)

# Define variables
X_train = train[["e_Dap_norm", "DeltaT_norm", "U_norm"]].copy()
X_test  = test[["e_Dap_norm", "DeltaT_norm", "U_norm"]].copy()

y_train = train["V_throat"]
y_test  = test["V_throat"]

# Interaction term
X_train["interaction"] = X_train["e_Dap_norm"] * X_train["U_norm"]
X_test["interaction"]  = X_test["e_Dap_norm"] * X_test["U_norm"]

# Add constant
X_train = sm.add_constant(X_train)
X_test  = sm.add_constant(X_test)

# Fit model
model = sm.OLS(y_train, X_train).fit()

print(model.summary())

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

r2_tr, rmse_tr, mae_tr = evaluate(y_train, y_pred_train)
r2_te, rmse_te, mae_te = evaluate(y_test, y_pred_test)

print("\nTrain Performance")
print(f"R2   = {r2_tr:.4f}")
print(f"RMSE = {rmse_tr:.4f}")
print(f"MAE  = {mae_tr:.4f}")

print("\nTest Performance")
print(f"R2   = {r2_te:.4f}")
print(f"RMSE = {rmse_te:.4f}")
print(f"MAE  = {mae_te:.4f}")

# Save path
save_path = os.path.dirname(file_path)

# Predicted vs Actual
plt.figure()
plt.scatter(y_test, y_pred_test)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)])
plt.xlabel("Actual V_throat")
plt.ylabel("Predicted V_throat")
plt.title("Predicted vs Actual")
plt.grid()
plt.savefig(os.path.join(save_path, "Predicted_vs_Actual.png"), dpi=300)
plt.show()

# Residual plot
residuals = y_test - y_pred_test

plt.figure()
plt.scatter(y_pred_test, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals")
plt.grid()
plt.savefig(os.path.join(save_path, "Residual_Plot.png"), dpi=300)
plt.show()

# Histogram
plt.figure()
plt.hist(residuals, bins=10)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.savefig(os.path.join(save_path, "Residual_Histogram.png"), dpi=300)
plt.show()
