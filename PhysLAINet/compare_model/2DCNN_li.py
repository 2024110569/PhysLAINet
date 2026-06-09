import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import rasterio
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 40
NUM_BANDS = 5
BATCH_SIZE = 3
EPOCHS = 1000
LR = 0.001
BETA = 0.98
PATIENCE = 30
EARLY_STOP_DELTA = 1e-4
IMAGE_ROOT = "../data/vi_images"
EXCEL_PATH = "../data/params.xlsx"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("vs", exist_ok=True)
BAND_FOLDERS = ["red", "green", "blue", "rededge", "nir"]
target_col = "LAI"
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

def load_all_images_to_memory(numbers, df):
    images = []
    lais = []
    for num in numbers:
        row = df[df["number"] == num].iloc[0]
        lai = float(row[target_col])
        bands = []
        for b in BAND_FOLDERS:
            path = os.path.join(IMAGE_ROOT, b, f"{num}.tif")
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            im = Image.fromarray(arr).resize((INPUT_SIZE, INPUT_SIZE))
            bands.append(np.array(im))
        img = np.stack(bands, axis=0).astype(np.float32)
        images.append(img)
        lais.append(lai)
    images = torch.tensor(np.array(images), dtype=torch.float32)
    lais = torch.tensor(np.array(lais), dtype=torch.float32)
    return images, lais

def load_numbers(p):
    with open(p) as f:
        return list(map(int, f.read().strip().split(",")))

train_nums = load_numbers("../data/dataset_idx/train_numbers.txt")
val_nums = load_numbers("../data/dataset_idx/valid_numbers.txt")
test_nums = load_numbers("../data/dataset_idx/test_numbers.txt")
df = pd.read_excel(EXCEL_PATH)
df["number"] = df["number"].astype(int)
print("Loading all images into memory...")
X_train, y_train = load_all_images_to_memory(train_nums, df)
X_val, y_val = load_all_images_to_memory(val_nums, df)
X_test, y_test = load_all_images_to_memory(test_nums, df)
X_trainval = torch.cat([X_train, X_val])
y_trainval = torch.cat([y_train, y_val])
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
trainval_dataset = TensorDataset(X_trainval, y_trainval)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
model = PaperCNN(5).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, BETA)
best_val_loss = float("inf")
counter = 0
save_path = "checkpoints/cnn_lai_model.pth"
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for img, y in train_loader:
        img, y = img.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)
    train_loss /= len(train_dataset)
    scheduler.step()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, y in val_loader:
            img, y = img.to(DEVICE), y.to(DEVICE)
            pred = model(img)
            val_loss += criterion(pred, y).item() * img.size(0)
    val_loss /= len(val_dataset)
    if val_loss < best_val_loss - EARLY_STOP_DELTA:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        counter += 1
    if counter >= PATIENCE:
        print(f"\n✅ Early stop at epoch {epoch+1}")
        break
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1:3d} | train={train_loss:.4f} | val={val_loss:.4f} | wait={counter}")
def predict(X_tensor):
    model.eval()
    with torch.no_grad():
        return model(X_tensor.to(DEVICE)).cpu().numpy()
model.load_state_dict(torch.load(save_path))
y_trainval_pred = predict(X_trainval)
y_test_pred = predict(X_test)
y_trainval_np = y_trainval.numpy()
y_test_np = y_test.numpy()

def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return r2, rmse

evaluate_model(y_trainval_np, y_trainval_pred, "Training + Validation Set")
evaluate_model(y_test_np, y_test_pred, "Test Set")

def plot_true_vs_pred(y_true, y_pred, dataset_name):
    font_size = 20
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=80)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='y=x')
    text_str = f'$R^2$ = {r2:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=font_size, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel(f"True {target_col}", fontsize=font_size)
    plt.ylabel(f"Predicted {target_col}", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(f"{dataset_name}", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size)
    plt.tight_layout()
    plt.savefig(f"vs/2DCNN_{target_col}_{dataset_name}.png", dpi=300)
    plt.show()

plot_true_vs_pred(y_trainval_np, y_trainval_pred, "Train_Valid")
plot_true_vs_pred(y_test_np, y_test_pred, "Test")