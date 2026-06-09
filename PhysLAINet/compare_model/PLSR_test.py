import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

MODEL_PATH = "checkpoints/plsr_LAI_model.pkl"
SCALER_PATH = "checkpoints/plsr_LAI_scaler.pkl"
DATA_PATH = "../data/2023/2023_yvmi_params.xlsx"
OUTPUT_DIR = "test_results_plsr"
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_plsr_test():
    if not os.path.exists(DATA_PATH):
        return
    data = pd.read_excel(DATA_PATH)
    data["number"] = data["number"].astype(int)
    data['plot_id'] = ((data['number'] - 1) % 30) + 1
    data['time_step'] = ((data['number'] - 1) // 30)
    data = data.sort_values(by=['plot_id', 'time_step']).reset_index(drop=True)
    feature_cols = ["red", "green", "blue", "rededge", "nir", "NDRE", "NDRE_std", "NDRE_smoothness", "NDRE_uniformity",
                    "NDRE_entropy", "Sowing_duration", "Acc_GDD", "Acc_Rain", "Recent_Rain_7d", "Diffuse_Ratio"]
    target_col = "LAI"
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return
    for col in feature_cols:
        lag_col_name = f"{col}_lag1"
        data[lag_col_name] = data.groupby('plot_id')[col].shift(1)
        data[lag_col_name] = data[lag_col_name].fillna(data[col])
    final_feature_order = feature_cols + [f"{c}_lag1" for c in feature_cols]
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    X_all = data[final_feature_order].values
    X_scaled = scaler.transform(X_all)
    y_pred_all = model.predict(X_scaled).flatten()
    data['Pred_LAI'] = y_pred_all
    eval_df = data[(data[target_col].notna()) & (data[target_col] != 0)].copy()
    y_true_raw = eval_df[target_col].values
    y_pred_raw = eval_df['Pred_LAI'].values
    r_squared = r2_score(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    print("\n" + "=" * 45)
    print(f"R²  : {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("=" * 45)
    output_df = eval_df[['number', 'Sowing_duration', target_col, 'Pred_LAI']].copy()
    output_df.rename(columns={target_col: 'True_LAI'}, inplace=True)
    output_df['Error'] = output_df['Pred_LAI'] - output_df['True_LAI']
    output_df['ABS_Error'] = np.abs(output_df['Error'])
    font_size = 20
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_raw, y_pred_raw, alpha=0.6, s=80, label='Predicted vs True')
    min_val = min(y_true_raw.min(), y_pred_raw.min())
    max_val = max(y_true_raw.max(), y_pred_raw.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, label='Perfect Prediction (y=x)')
    text_str = f'$R^2$ = {r_squared:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(
        x=0.05, y=0.95,
        s=text_str,
        transform=plt.gca().transAxes,
        fontsize=font_size-5,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Predicted LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Test B", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size-5)
    plt.tight_layout()
    img_save_path = os.path.join(OUTPUT_DIR, "plsr_2023_comparison_raw.png")
    plt.savefig(img_save_path, dpi=330)
    plt.show()


if __name__ == "__main__":
    run_plsr_test()