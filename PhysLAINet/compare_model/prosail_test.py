import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

MODEL_PATH = "checkpoints/best_finetune.pth"
SCALER_PATH = "checkpoints/scaler.pkl"
DATA_PATH = "../data/test.xlsx"
OUTPUT_DIR = "test_results_dnn"

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PROSAIL_DNN(nn.Module):
    def __init__(self, in_dim=11):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512), nn.Softplus(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.Softplus(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.Softplus(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Softplus(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

def run_dnn_test():
    if not os.path.exists(DATA_PATH):
        print(f"No data: {DATA_PATH}")
        return
    data = pd.read_excel(DATA_PATH)
    if "number" in data.columns:
        data["number"] = data["number"].astype(int)
    feature_cols = ["blue","green","red","rededge","nir","NDVI","NDRE","RVI","DVI","EVI","OSAVI"]
    target_col = "LAI"
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("No model or scaler")
        return
    scaler = joblib.load(SCALER_PATH)
    model = PROSAIL_DNN(in_dim=len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    X = data[feature_cols].values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().squeeze()
    data['Pred_LAI'] = y_pred
    eval_df = data[(data[target_col].notna()) & (data[target_col] != 0)].copy()
    y_true = eval_df[target_col].values
    y_pred = eval_df['Pred_LAI'].values
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("\n" + "=" * 45)
    print(f"LAI_MEAN: {y_true.mean():.4f} | STD: {y_true.std():.4f}")
    print("-" * 45)
    print(f"R²      : {r2:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"MAE     : {mae:.4f}")
    print("=" * 45)
    output_df = eval_df[['number', target_col, 'Pred_LAI']].copy()
    output_df.rename(columns={target_col: 'True_LAI'}, inplace=True)
    output_df['Error'] = output_df['Pred_LAI'] - output_df['True_LAI']
    output_df['ABS_Error'] = np.abs(output_df['Error'])
    save_excel = os.path.join(OUTPUT_DIR, "dnn_test_predictions.xlsx")
    output_df.to_excel(save_excel, index=False)
    font_size = 20
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=80, label='Predicted vs True')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    text_str = f'$R^2$ = {r2:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=font_size-5,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Predicted LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Test Set", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size-5)
    plt.tight_layout()
    save_img = os.path.join(OUTPUT_DIR, "dnn_test_comparison.png")
    plt.savefig(save_img, dpi=330)
    plt.show()
    print(f"Test results saved: {save_excel}")
    print(f"VS results saved: {save_img}")

if __name__ == "__main__":
    run_dnn_test()