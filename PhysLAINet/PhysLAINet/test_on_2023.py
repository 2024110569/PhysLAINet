import torch
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from physlainet import PhysLAINet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/physlainet.pth"
SCALER_PATH = "checkpoints/physlainet.joblib"
DATA_PATH = "../data/2023/2023_yvmi_params.xlsx"
OUTPUT_DIR = "test_results"
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
os.makedirs(OUTPUT_DIR, exist_ok=True)
@torch.no_grad()
def analyze_branch_contribution(model, X_t, y_true_raw):
    model.eval()
    X_t = X_t.to(DEVICE)
    p_final, p_phys, p_grow = model(X_t, return_3_val=True)
    p_final_np = p_final.cpu().numpy().flatten()
    p_phys_np = p_phys.cpu().numpy().flatten()
    p_grow_np = p_grow.cpu().numpy().flatten()
    r2_final = r2_score(y_true_raw, p_final_np)
    r2_phys = r2_score(y_true_raw, p_phys_np)
    r2_grow = r2_score(y_true_raw, p_grow_np)
    rmse_final = np.sqrt(mean_squared_error(y_true_raw, p_final_np))
    mae_phys = np.mean(np.abs(p_phys_np - y_true_raw))
    mae_grow = np.mean(np.abs(p_grow_np - y_true_raw))
    _, _, _, alpha = model(X_t, return_attn=True)
    alpha_mean = torch.sigmoid(alpha).mean().item()
    rmse_phys = np.sqrt(mean_squared_error(y_true_raw, p_phys))
    rmse_grow = np.sqrt(mean_squared_error(y_true_raw, p_grow))
    print("\n" + "=" * 40)
    print(f"alpha: {alpha_mean:.4f}")
    print("-" * 40)
    print(f"final R²: {r2_final:.4f} | RMSE: {rmse_final:.4f}")
    print(f"phys R²: {r2_phys:.4f} | MAE: {mae_phys:.4f} | RMSE: {rmse_phys:.4f}")
    print(f"pheno R²: {r2_grow:.4f} | MAE: {mae_grow:.4f} | RMSE: {rmse_grow:.4f}")
    print("=" * 40)
    font_size = 20
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_raw, p_phys_np, alpha=0.5, label=f'R²={r2_phys:.4f}\nRMSE={rmse_phys:.4f}', color='blue')
    lims = [0, max(y_true_raw.max(), p_phys_np.max()) + 0.5]
    plt.plot(lims, lims, '--', color='gray')
    plt.title("Physical Branch (Raw Scale)", fontsize=font_size)
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Pred LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_raw, p_grow_np, alpha=0.5, label=f'R²={r2_grow:.4f}\nRMSE={rmse_grow:.4f}', color='green')
    lims = [0, max(y_true_raw.max(), p_grow_np.max()) + 0.5]
    plt.plot(lims, lims, '--', color='gray')
    plt.title("Growth Trend (Raw Scale)", fontsize=font_size)
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Pred LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Branch_Performance_Raw.png", dpi=330)
    plt.close()

def run_test():
    if not os.path.exists(DATA_PATH):
        print(f"File not found {DATA_PATH}")
        return
    data = pd.read_excel(DATA_PATH)
    data["number"] = data["number"].astype(int)
    data['plot_id'] = ((data['number'] - 1) % 30) + 1
    data['time_step'] = ((data['number'] - 1) // 30)
    data = data.sort_values(by=['plot_id', 'time_step']).reset_index(drop=True)
    feature_cols = ["red", "green", "blue", "rededge", "nir", "NDRE", "NDRE_std", "NDRE_smoothness", "NDRE_uniformity",
                    "NDRE_entropy", "Sowing_duration", "Acc_GDD", "Acc_Rain", "Recent_Rain_7d", "Diffuse_Ratio"]
    target_col = "LAI"
    lag_features = []
    for col in feature_cols:
        lag_col_name = f"{col}_lag1"
        data[lag_col_name] = data.groupby('plot_id')[col].shift(1)
        data[lag_col_name] = data[lag_col_name].fillna(data[col])
        lag_features.append(lag_col_name)
    final_feature_order = feature_cols + lag_features
    scaler = joblib.load(SCALER_PATH)
    model = PhysLAINet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    eval_df = data[(data[target_col].notna()) & (data[target_col] != 0)].copy()
    X_eval = eval_df[final_feature_order].values
    X_eval_scaled = scaler.transform(X_eval)
    X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32).to(DEVICE)
    y_true_raw = eval_df[target_col].values
    with torch.no_grad():
        y_pred, opt_attns, env_attns, _ = model(X_eval_tensor, return_attn=True)
        y_pred = y_pred.cpu().numpy().flatten()
    eval_df['Pred_LAI'] = y_pred
    r_squared = r2_score(y_true_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred))
    print(f"\nTrue LAI: Mean = {y_true_raw.mean():.4f}, Std = {y_true_raw.std():.4f}")
    print(f"Pred LAI: Mean = {y_pred.mean():.4f}, Std = {y_pred.std():.4f}")
    print("\n" + "=" * 40)
    print(f"2023 test result")
    print(f"sample numbers: {len(eval_df)}")
    print(f"R²  : {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("=" * 40)
    output_df = eval_df[['number', 'Sowing_duration', target_col, 'Pred_LAI']].copy()
    output_df.rename(columns={target_col: 'True_LAI'}, inplace=True)
    output_df['Absolute_Error'] = output_df['Pred_LAI'] - output_df['True_LAI']
    font_size = 20
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_raw, y_pred, alpha=0.6, s=80, label='Predicted vs True')
    min_val = min(y_true_raw.min(), y_pred.min())
    max_val = max(y_true_raw.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    text_str = f'$R^2$ = {r_squared:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(
        x=0.05, y=0.95,
        s=text_str,
        transform=plt.gca().transAxes,
        fontsize=font_size - 5,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Predicted LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Test A", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size - 5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2023_comparison.png", dpi=300)
    plt.show()
    analyze_branch_contribution(model, X_eval_tensor, y_true_raw)

if __name__ == "__main__":
    run_test()
