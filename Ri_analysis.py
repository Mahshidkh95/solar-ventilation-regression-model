import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Load data
# =========================
file_path = "SVS_A_30deg_dimensionless.xlsx"
df = pd.read_excel(file_path)

# =========================
# Constants
# =========================
g = 9.81
L = 2.14  # m
beta = 1 / 300  # standard approximation

# =========================
# Calculate Ri
# =========================
df["Ri"] = (g * beta * df["ΔT = T_col - T_amb (°C)"] * L) / (df["Wind speed, U (m/s)"]**2)

# =========================
# Handle U = 0
# =========================
df.loc[df["Wind speed, U (m/s)"] == 0, "Ri"] = np.nan

# =========================
# Classify flow regime
# =========================
def classify_ri(ri):
    if pd.isna(ri):
        return "Pure Buoyancy"
    elif ri > 10:
        return "Buoyancy Dominated"
    elif ri < 0.1:
        return "Wind Dominated"
    else:
        return "Mixed Regime"

df["Flow Regime"] = df["Ri"].apply(classify_ri)

# =========================
# Save Excel (same folder)
# =========================
save_path = os.path.dirname(file_path)
excel_out = os.path.join(save_path, "SVS_A_with_Ri_and_Regime.xlsx")

df.to_excel(excel_out, index=False)

# =========================
# Plot (professional style)
# =========================
plt.figure()

# remove NaN Ri for plotting
df_plot = df.dropna(subset=["Ri"])

for regime in df_plot["Flow Regime"].unique():
    subset = df_plot[df_plot["Flow Regime"] == regime]
    plt.scatter(subset["Ri"], subset["V_throat"], label=regime)

plt.xscale("log")

plt.xlabel("Richardson Number (Ri)")
plt.ylabel("V_throat (m/s)")
plt.title("Ri vs V_throat (SVS-A, 30°)")
plt.legend()
plt.grid()

# save in same folder
plot_out = os.path.join(save_path, "Ri_vs_Vthroat.png")
plt.savefig(plot_out, dpi=300)

plt.show()

# =========================
# Print Ri range
# =========================
print("\nRi range:")
print("Min =", df["Ri"].min())
print("Max =", df["Ri"].max())

print("\nDone. Files saved.")
