import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 9945  # As paper

params = {
    "N": np.random.uniform(1.0, 2.0, n_samples),
    "Cab": np.random.uniform(30, 90, n_samples),
    "Car": np.random.uniform(5, 18, n_samples),
    "Cw": np.random.uniform(0.009, 0.03, n_samples),
    "Cm": np.random.uniform(0.003, 0.008, n_samples),
    "LAI": np.random.uniform(0.0, 7.0, n_samples),
    "lidfa": np.random.uniform(20, 60, n_samples),
    "hotspot": np.random.uniform(0.05, 1.0, n_samples),
    "psoil": np.random.uniform(0.1, 0.8, n_samples),
    "tts": np.full(n_samples, 22.0),
    "psi": np.full(n_samples, 162.0),
}

# Satisfied to Train Set A. Blue:475, Green:560, Red:668, Rededge:717, NIR:842
def simulate_corn_reflectance(params):
    LAI = params["LAI"]
    blue = 0.015 + 0.08 * np.exp(-LAI / 1.5) + np.random.normal(0, 0.0025, n_samples)
    green = 0.025 + 0.11 * np.exp(-LAI / 2.0) + np.random.normal(0, 0.0025, n_samples)
    red = 0.030 + 0.16 * np.exp(-LAI / 2.2) + np.random.normal(0, 0.0025, n_samples)
    rededge = 0.130 + 0.32 * (1 - np.exp(-LAI / 3.0)) + np.random.normal(0, 0.0025, n_samples)
    nir = 0.270 + 0.58 * (1 - np.exp(-LAI / 4.0)) + np.random.normal(0, 0.0030, n_samples)
    blue = np.clip(blue, 0.01, 0.40)
    green = np.clip(green, 0.02, 0.50)
    red = np.clip(red, 0.03, 0.45)
    rededge = np.clip(rededge, 0.10, 0.70)
    nir = np.clip(nir, 0.25, 0.90)
    return blue, green, red, rededge, nir


blue, green, red, rededge, nir = simulate_corn_reflectance(params)

NDVI = (nir - red) / (nir + red + 1e-8)
NDRE = (nir - rededge) / (nir + rededge + 1e-8)
RVI = nir / (red + 1e-8)
DVI = nir - red
EVI = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
OSAVI = 1.16 * (nir - red) / (nir + red + 0.16 + 1e-8)

sim_df = pd.DataFrame({
    "blue": blue, "green": green, "red": red, "rededge": rededge, "nir": nir,
    "NDVI": NDVI, "NDRE": NDRE, "RVI": RVI, "DVI": DVI, "EVI": EVI, "OSAVI": OSAVI,
    "LAI": params["LAI"]
})

sim_df.to_csv("prosail_simulation_9945.csv", index=False)
print("Done.")