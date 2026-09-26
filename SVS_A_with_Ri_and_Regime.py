import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# Load data
# =========================
file_path = "SVS_A_with_Ri_and_Regime.xlsx"
df = pd.read_excel(file_path)

# =========================
# Clean column names
# =========================
df.columns = df.columns.str.strip()

# =========================
# Rename columns (exact match with your Excel)
# =========================
df = df.rename(columns={
    "V_throat (m/s)": "V_throat",
    "Wind speed, U (m/s)": "U",
    "ΔT = T_col - T_amb (°C)": "DeltaT",
    "e / D_ap": "eD"
})

# =========================
# Remove pure buoyancy (U=0)
# =========================
df = df[df["U"] > 0].copy()

# =========================
# Check regime names (IMPORTANT)
# =========================
print("\nAvailable regimes:")
print(df["Flow Regime"].unique())

# =========================
# Regression function
# =========================
def run_model(data, name):

    print("\n==============================")
    print(f"REGIME: {name}")
    print("==============================")

    # Features
    X = data[["eD", "DeltaT", "U"]].copy()
    X["interaction"] = X["eD"] * X["U"]

    y = data["V_throat"]

    # Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Add constant
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit model (ONLY on train)
    model = sm.OLS(y_train, X_train).fit()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # R²
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Print results
    print(f"\nTrain R² = {r2_train:.4f}")
    print(f"Test  R² = {r2_test:.4f}")

    # Equation
    c = model.params

    print("\nRegression Equation:")
    print(
        f"V_throat = {c['const']:.4f} "
        f"+ {c['eD']:.4f}(e/D_ap) "
        f"+ {c['DeltaT']:.4f}(DeltaT) "
        f"+ {c['U']:.4f}(U) "
        f"+ {c['interaction']:.4f}(U × e/D_ap)"
    )

    return r2_train, r2_test, model


# =========================
# Split regimes (BASED ON YOUR FILE)
# =========================
mixed = df[df["Flow Regime"] == "Mixed Regime"]
wind  = df[df["Flow Regime"] == "Wind Dominated"]

# =========================
# Run models
# =========================
r2_mixed_train, r2_mixed_test, model_mixed = run_model(mixed, "Mixed Regime")
r2_wind_train,  r2_wind_test,  model_wind  = run_model(wind, "Wind Dominated")

print("\nDone ✔")
