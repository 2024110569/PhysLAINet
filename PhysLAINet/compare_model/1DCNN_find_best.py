import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("vs", exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
data = pd.read_excel("../data/params.xlsx")
data["number"] = data["number"].astype(int)
data['plot_id'] = ((data['number'] - 1) % 80) + 1
data['time_step'] = ((data['number'] - 1) // 80)
data = data.sort_values(by=['plot_id', 'time_step']).reset_index(drop=True)
feature_cols = [
    "red", "green", "blue", "rededge", "nir", "NDRE", "NDRE_std",
    "NDRE_smoothness", "NDRE_uniformity", "NDRE_entropy",
    "Sowing_duration", "Acc_GDD", "Acc_Rain", "Recent_Rain_7d", "Diffuse_Ratio"
]
target_col = "LAI"
lag_features = []
for col in feature_cols:
    lag_col_name = f"{col}_lag1"
    data[lag_col_name] = data.groupby('plot_id')[col].shift(1)
    data[lag_col_name] = data[lag_col_name].fillna(data[col])
    lag_features.append(lag_col_name)
extended_feature_cols = feature_cols + lag_features
def load_numbers(file_path):
    with open(file_path, "r") as f:
        return list(map(int, f.read().strip().split(",")))
train_nums = load_numbers("../data/dataset_idx/train_numbers.txt")
val_nums = load_numbers("../data/dataset_idx/valid_numbers.txt")
test_nums = load_numbers("../data/dataset_idx/test_numbers.txt")
train_data = data[data["number"].isin(train_nums)]
val_data = data[data["number"].isin(val_nums)]
test_data = data[data["number"].isin(test_nums)]
X_train, y_train = train_data[extended_feature_cols].values, train_data[target_col].values
X_val, y_val = val_data[extended_feature_cols].values, val_data[target_col].values
X_test, y_test = test_data[extended_feature_cols].values, test_data[target_col].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"Train set: {len(X_train)} samples.")
print(f"Valid set: {len(X_val)} samples.")
print(f"Test set: {len(X_test)} samples.")
class LAIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
BATCH_SIZE = 16
train_dataset = LAIDataset(X_train_scaled, y_train)
val_dataset = LAIDataset(X_val_scaled, y_val)
test_dataset = LAIDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LAI(input_dim=30).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
EPOCHS = 100
best_val_loss = float('inf')
model_save_path = "checkpoints/cnn_lai_model.pth"
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
    train_loss /= len(train_dataset)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            val_loss += loss.item() * X.size(0)
    val_loss /= len(val_dataset)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
model.load_state_dict(torch.load(model_save_path))
model.eval()
def predict(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X_tensor).cpu().numpy()
y_train_pred = predict(X_train_scaled)
y_val_pred = predict(X_val_scaled)
y_trainval_pred = predict(np.concatenate([X_train_scaled, X_val_scaled]))
y_test_pred = predict(X_test_scaled)
y_trainval = np.concatenate([y_train, y_val])
def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return r2, rmse
evaluate_model(y_trainval, y_trainval_pred, "Training + Validation Set")
evaluate_model(y_test, y_test_pred, "Test Set")
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
    plt.savefig(f"vs/CNN_{target_col}_{dataset_name}.png", dpi=300)
    plt.show()
plot_true_vs_pred(y_trainval, y_trainval_pred, "Train_Valid")
plot_true_vs_pred(y_test, y_test_pred, "Test")
joblib.dump(scaler, "checkpoints/cnn_lai_scaler.pkl")