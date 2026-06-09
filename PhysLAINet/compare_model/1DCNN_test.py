import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

MODEL_PATH = "checkpoints/cnn_lai_model.pth"
SCALER_PATH = "checkpoints/cnn_lai_scaler.pkl"
DATA_PATH = "../data/2023/2023_yvmi_params.xlsx"
OUTPUT_DIR = "test_results_cnn"
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CNN_LAI(nn.Module):
    def __init__(self, input_dim=30):
        super(CNN_LAI, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

def run_cnn_test():
    if not os.path.exists(DATA_PATH):
        return
    data = pd.read_excel(DATA_PATH)
    data["number"] = data["number"].astype(int)
    data['plot_id'] = ((data['number'] - 1) % 30) + 1
    data['time_step'] = ((data['number'] - 1) // 30)
    data = data.sort_values(by=['plot_id', 'time_step']).reset_index(drop=True)
    feature_cols = ["red", "green", "blue", "rededge", "nir", "NDRE", "NDRE_std", "NDRE_smoothness",
                    "NDRE_uniformity", "NDRE_entropy", "Sowing_duration", "Acc_GDD", "Acc_Rain",
                    "Recent_Rain_7d", "Diffuse_Ratio"]
    target_col = "LAI"
    lag_features = []
    for col in feature_cols:
        lag_col_name = f"{col}_lag1"
        data[lag_col_name] = data.groupby('plot_id')[col].shift(1)
        data[lag_col_name] = data[lag_col_name].fillna(data[col])
        lag_features.append(lag_col_name)
    extended_feature_cols = feature_cols + lag_features
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("no scaler")
        return
    scaler = joblib.load(SCALER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LAI(input_dim=30).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    X_all = data[extended_feature_cols].values
    X_scaled = scaler.transform(X_all)
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_pred_all = model(X_tensor).cpu().numpy()
    data['Pred_LAI'] = y_pred_all
    eval_df = data[(data[target_col].notna()) & (data[target_col] != 0)].copy()
    y_true_raw = eval_df[target_col].values
    y_pred_raw = eval_df['Pred_LAI'].values
    r_squared = r2_score(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    print("\n" + "=" * 45)
    print(f"2023")
    print(f"sample numbers {len(eval_df)}")
    print(f"mean {y_true_raw.mean():.4f} | std {y_true_raw.std():.4f}")
    print("-" * 45)
    print(f"R²  : {r_squared:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("=" * 45)
    output_df = eval_df[['number', 'Sowing_duration', target_col, 'Pred_LAI']].copy()
    output_df.rename(columns={target_col: 'True_LAI'}, inplace=True)
    output_df['Error'] = output_df['Pred_LAI'] - output_df['True_LAI']
    output_df['ABS_Error'] = np.abs(output_df['Error'])
    output_df.to_excel(save_path, index=False)
    font_size = 20
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_raw, y_pred_raw, alpha=0.6, s=80, label='Predicted vs True')
    min_val = min(y_true_raw.min(), y_pred_raw.min())
    max_val = max(y_true_raw.max(), y_pred_raw.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    text_str = f'$R^2$ = {r_squared:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=font_size - 5, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Predicted LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Test B", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size - 5)
    plt.tight_layout()
    img_save_path = os.path.join(OUTPUT_DIR, "cnn_2023_comparison.png")
    plt.savefig(img_save_path, dpi=330)
    plt.show()


if __name__ == "__main__":
    run_cnn_test()