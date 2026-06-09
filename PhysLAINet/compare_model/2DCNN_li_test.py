import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import rasterio

MODEL_PATH = "checkpoints/cnn_lai_model.pth"
DATA_PATH = "../data/2023/2023_yvmi_params.xlsx"
IMAGE_DIR = "../data/2023"
OUTPUT_DIR = "test_results_2dcnn"
INPUT_SIZE = 40
NUM_BANDS = 5
BAND_FOLDERS = ["red", "green", "blue", "rededge", "nir"]
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
os.makedirs(OUTPUT_DIR, exist_ok=True)
class PaperCNN(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 10, 3, padding=1), nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=1), nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, 3, padding=1), nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
            nn.Conv2d(6, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(),
            nn.Conv2d(3, 1, 3, padding=1), nn.BatchNorm2d(1), nn.ReLU(),
        )
        self.fc = nn.Linear(1 * INPUT_SIZE * INPUT_SIZE, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze()

def center_crop(img, crop_size):
    h, w = img.shape[-2:]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[..., start_h:start_h+crop_size, start_w:start_w+crop_size]

def load_single_image(num):
    bands = []
    for b in BAND_FOLDERS:
        img_path = os.path.join(IMAGE_DIR, b, f"{num}.tif")
        with rasterio.open(img_path) as src:
            arr = src.read(1).astype(np.float32)
        bands.append(arr)
    img = np.stack(bands, axis=0).astype(np.float32)
    img = center_crop(img, INPUT_SIZE)
    return img

def run_2dcnn_test():
    if not os.path.exists(DATA_PATH):
        return
    data = pd.read_excel(DATA_PATH)
    data["number"] = data["number"].astype(int)
    numbers = data["number"].values
    target_col = "LAI"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperCNN(in_channels=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    images = []
    for num in numbers:
        img = load_single_image(num)
        images.append(img)
    images = np.array(images).astype(np.float32)
    with torch.no_grad():
        X_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy()
    data['Pred_LAI'] = y_pred
    eval_df = data[(data[target_col].notna()) & (data[target_col] != 0)].copy()
    y_true = eval_df[target_col].values
    y_pred = eval_df['Pred_LAI'].values
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("\n" + "=" * 50)
    print(f"R²      : {r2:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print("=" * 50)
    output_df = eval_df[['number', target_col, 'Pred_LAI']].copy()
    output_df.rename(columns={target_col: 'True_LAI'}, inplace=True)
    output_df['Error'] = output_df['Pred_LAI'] - output_df['True_LAI']
    output_df['ABS_Error'] = np.abs(output_df['Error'])
    excel_path = os.path.join(OUTPUT_DIR, "2dcnn_2023_predictions.xlsx")
    output_df.to_excel(excel_path, index=False)
    font_size = 20
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=80)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    text_str = f'$R^2$ = {r2:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=15, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Predicted LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("Test B", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=15)
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "2dcnn_2023_comparison.png")
    plt.savefig(img_path, dpi=330)
    plt.show()

if __name__ == "__main__":
    run_2dcnn_test()