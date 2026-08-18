import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
import os
import seaborn as sns
import joblib


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

def load_numbers(file_path):
    with open(file_path, "r") as f:
        return list(map(int, f.read().strip().split(",")))

def trapz_onnx(y, x=None, dim=-1):
    y_sum = y.sum(dim=dim) - 0.5 * (y.select(dim, 0) + y.select(dim, -1))
    if x is None:
        return y_sum
    if x.dim() == 1:
        x = x.reshape((1,) * (y.dim() - 1) + (-1,))
    dx = x.select(dim, 1) - x.select(dim, 0)
    return y_sum * dx

class RELU_ONNX(nn.Module):
    def forward(self, x):
        return torch.relu(x)

class PhysLAINet(nn.Module):
    def __init__(self, input_dim=30, n_steps=30):
        super().__init__()
        self.opt_dim = 20
        self.env_dim = 10
        self.n_steps = n_steps
        self.opt_attn = nn.Sequential(
            nn.Linear(self.opt_dim, 32),
            nn.LayerNorm(32),
            RELU_ONNX(),
            nn.Dropout(0.1),
            nn.Linear(32, self.opt_dim),
            nn.Sigmoid()
        )
        self.env_attn = nn.Sequential(
            nn.Linear(self.env_dim, 16),
            nn.LayerNorm(16),
            RELU_ONNX(),
            nn.Linear(16, self.env_dim),
            nn.Sigmoid()
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(self.env_dim, self.opt_dim),
            nn.Sigmoid()
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            RELU_ONNX(),
            nn.Linear(64, 64)
        )
        self.res_shortcut = nn.Linear(input_dim, 64)
        self.structure_head = nn.Sequential(
            nn.Linear(64, 32), RELU_ONNX(), nn.Linear(32, 3)
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(64, 32), RELU_ONNX(), nn.Linear(32, 3)
        )
        self.growth_params_net = nn.Sequential(
            nn.Linear(2, 16), RELU_ONNX(), nn.Linear(16, 3)
        )
        self.pheno_scale_head = nn.Sequential(
            nn.Linear(64, 32), RELU_ONNX(), nn.Linear(32, 1)
        )
        self.log_var_phys = nn.Parameter(torch.zeros(1))
        self.log_var_grow = nn.Parameter(torch.zeros(1))
        self.fusion_logit = nn.Sequential(
            nn.Linear(1, 16),
            RELU_ONNX(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.gdd_scale_log = nn.Parameter(torch.tensor([4.6]))
        self.clamp_val = nn.Parameter(torch.tensor([20.0]))
        self.tm_base = nn.Parameter(torch.tensor([500.0]))
        self.tm_range = nn.Parameter(torch.tensor([1500.0]))
        self.z_base = nn.Parameter(torch.tensor([0.02]))
        self.sat_alpha = nn.Parameter(torch.tensor([0.1]))
        self.sat_beta = nn.Parameter(torch.tensor([0.001]))

    def forward(self, x, return_attn=False, return_3_val=False):
        x_opt = torch.cat([x[:, :10], x[:, 15:25]], dim=1)
        x_env = torch.cat([x[:, 10:15], x[:, 25:30]], dim=1)
        acc_gdd_curr = x[:, 26:27]
        w_opt = self.opt_attn(x_opt)
        w_env = self.env_attn(x_env)
        gate = self.cross_gate(x_env * w_env)
        f_in = torch.cat([x_opt * w_opt * gate, x_env * w_env], dim=1)
        feat = RELU_ONNX()(self.feature_extractor(f_in) + self.res_shortcut(f_in))
        p_struct = self.structure_head(feat)
        p_intent = self.intensity_head(feat)
        beta = torch.sigmoid(self.pheno_scale_head(feat))
        X0 = torch.sigmoid(p_struct[:, 0:1]) + beta * torch.tanh(acc_gdd_curr)
        k_width = -F.softplus(p_struct[:, 1:2])
        b_skew = torch.clamp(p_struct[:, 2:3], -2.0, 2.0)
        rho = F.softplus(p_intent[:, 0:1])
        S_raw = torch.sigmoid(p_intent[:, 1:2])
        sat = torch.sigmoid(p_intent[:, 2:3]) * self.sat_alpha + self.sat_beta
        mag_term = ((rho * S_raw) / (1.0 + sat * (rho * S_raw) ** 2) + self.z_base)
        z_space = torch.linspace(0, 1, self.n_steps).to(x.device)
        dist = z_space.unsqueeze(0) - X0
        exponent = k_width * (dist ** 2) + b_skew * (dist ** 3)
        lad_z = mag_term * torch.exp(exponent)
        instant_lai = trapz_onnx(lad_z, z_space, dim=1).unsqueeze(1)
        acc_gdd_info = torch.cat([x[:, 11:12], x[:, 26:27]], dim=1)
        g_p = self.growth_params_net(acc_gdd_info)
        l_max = F.softplus(g_p[:, 0:1]) * 8.0
        k_grow = torch.clamp(F.softplus(g_p[:, 1:2]), 0.01, 0.5)
        tm = torch.sigmoid(g_p[:, 2:3]) * self.tm_range + self.tm_base
        scale_factor = torch.exp(self.gdd_scale_log)
        diff_raw = acc_gdd_curr - tm
        diff_scaled = diff_raw / scale_factor
        c_val = torch.abs(self.clamp_val)
        diff_final = torch.clamp(diff_scaled, -c_val, c_val)
        lai_growth_trend = l_max / (1 + torch.exp(-k_grow * diff_final))
        alpha = self.fusion_logit(acc_gdd_curr)
        final_lai = alpha * instant_lai + (1 - alpha) * lai_growth_trend
        if return_attn:
            return final_lai, w_opt, w_env, alpha
        if return_3_val:
            return final_lai, instant_lai, lai_growth_trend
        return final_lai

class CombinedUncertaintyWeightedLoss(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, p_final, p_phys, p_grow, y_true, log_var_phys, log_var_grow):
        sample_weights = torch.pow(y_true + 1.0, self.alpha)
        sample_weights = sample_weights / sample_weights.mean()
        def weighted_huber(pred, target):
            loss_elementwise = F.huber_loss(pred, target, delta=1.0, reduction='none')
            return torch.mean(loss_elementwise * sample_weights)
        l_final = weighted_huber(p_final, y_true)
        l_phys = weighted_huber(p_phys, y_true)
        l_grow = weighted_huber(p_grow, y_true)
        inv_phys = torch.exp(-log_var_phys)
        loss_p = inv_phys * (l_final + l_phys) + log_var_phys
        inv_grow = torch.exp(-log_var_grow)
        loss_g = inv_grow * (l_final + l_grow) + log_var_grow
        return torch.mean(loss_p + loss_g)
@torch.no_grad()
def evaluate_and_get_preds(model, X_t, y_true, dataset_name):
    model.eval()
    X_t = X_t.to(DEVICE)
    preds, opt_attns, env_attns, _ = model(X_t, return_attn=True)
    preds = preds.cpu().numpy().flatten()
    opt_attns = opt_attns.cpu().numpy()
    env_attns = env_attns.cpu().numpy()
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    print(f"{dataset_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return preds, opt_attns, env_attns

def plot_true_vs_pred(y_true, y_pred, dataset_name):
    font_size=20
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=80, label='Predicted vs True')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction (y=x)')
    text_str = f'$R^2$ = {r2:.4f}\n$RMSE$ = {rmse:.4f}'
    plt.text(
        x=0.05, y=0.95,
        s=text_str,
        transform=plt.gca().transAxes,
        fontsize=font_size,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel(f"True {target_col}", fontsize=font_size)
    plt.ylabel(f"Predicted {target_col}", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(f"{dataset_name}", fontsize=font_size, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=font_size)
    plt.tight_layout()
    os.makedirs("vs", exist_ok=True)
    plt.savefig(f"vs/{target_col}_{dataset_name}.png", dpi=330)
    plt.close()


@torch.no_grad()
def analyze_branch_contribution(model, X_t, y_true):
    model.eval()
    X_t = X_t.to(DEVICE)
    y_true = y_true.cpu().numpy().flatten()
    p_final, p_phys, p_grow = model(X_t, return_3_val=True)
    p_final = p_final.cpu().numpy().flatten()
    p_phys = p_phys.cpu().numpy().flatten()
    p_grow = p_grow.cpu().numpy().flatten()
    final_lai, w_opt, w_env, alpha = model(X_t, return_attn=True)
    alpha = torch.sigmoid(alpha).mean().item()
    r2_final = r2_score(y_true, p_final)
    r2_phys = r2_score(y_true, p_phys)
    r2_grow = r2_score(y_true, p_grow)
    mae_phys = np.mean(np.abs(p_phys - y_true))
    mae_grow = np.mean(np.abs(p_grow - y_true))
    rmse_phys = np.sqrt(mean_squared_error(y_true, p_phys))
    rmse_grow = np.sqrt(mean_squared_error(y_true, p_grow))

    print("=" * 30)
    print(f"Alpha: {alpha:.4f} phys vs {1 - alpha:.4f} pheno")
    print("-" * 30)
    print(f"final R²: {r2_final:.4f}")
    print(f"phys R²: {r2_phys:.4f} | MAE: {mae_phys:.4f} | RMSE: {rmse_phys:.4f}")
    print(f"pheno R²: {r2_grow:.4f} | MAE: {mae_grow:.4f} | RMSE: {rmse_grow:.4f}")
    print("=" * 30)
    font_size = 20
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, p_phys, alpha=0.5, label=f'R²={r2_phys:.4f}\nRMSE={rmse_phys:.4f}', color='blue')
    lims = [0, max(y_true.max(), p_phys.max()) + 0.5]
    plt.plot(lims, lims, '--', color='gray')
    plt.title("Physical Branch Performance", fontsize=font_size)
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Pred LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, p_grow, alpha=0.5, label=f'R²={r2_grow:.4f}\nRMSE={rmse_grow:.4f}', color='green')
    lims = [0, max(y_true.max(), p_phys.max()) + 0.5]
    plt.plot(lims, lims, '--', color='gray')
    plt.title("Growth Trend Performance", fontsize=font_size)
    plt.xlabel("True LAI", fontsize=font_size)
    plt.ylabel("Pred LAI", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.savefig(f"results/Branchs_{target_col}.png", dpi=330)
    plt.show()

if __name__ == '__main__':
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_excel("../data/params.xlsx")
    data["number"] = data["number"].astype(int)
    data['plot_id'] = ((data['number'] - 1) % 80) + 1
    data['time_step'] = ((data['number'] - 1) // 80)
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
    extended_feature_cols = feature_cols + lag_features
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
    def to_tensor(X, y):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
    X_train_t, y_train_t = to_tensor(X_train_scaled, y_train)
    X_val_t, y_val_t = to_tensor(X_val_scaled, y_val)
    X_test_t, y_test_t = to_tensor(X_test_scaled, y_test)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
    model = PhysLAINet().to(DEVICE)
    criterion = CombinedUncertaintyWeightedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        min_lr=1e-6
    )
    EPOCHS = 1000
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience, patience_counter = 100, 0
    print("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            p_final, p_phys, p_grow = model(batch_x, return_3_val=True)
            loss = criterion(p_final, p_phys, p_grow, batch_y, model.log_var_phys, model.log_var_grow)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            p_final, p_phys, p_grow = model(X_val_t.to(DEVICE), return_3_val=True)
            val_loss = criterion(p_final, p_phys, p_grow, y_val_t.to(DEVICE), model.log_var_phys, model.log_var_grow)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch + 1}: Train Loss = {total_train_loss / len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

    model.load_state_dict(best_model_weights)
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "physlainet.pth")
    torch.save(model.state_dict(), model_path)
    print(f"pth saved: {model_path}")
    scaler_path = os.path.join(save_dir, "physlainet.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"scaler saved: {scaler_path}")
    print("\n--- result ---")
    y_train_pred, train_opt_attns, train_env_attns = evaluate_and_get_preds(model, X_train_t, y_train, "Train")
    y_val_pred, val_opt_attns, val_env_attns = evaluate_and_get_preds(model, X_val_t, y_val, "Valid")
    y_test_pred, test_opt_attns, test_env_attns = evaluate_and_get_preds(model, X_test_t, y_test, "Test")
    plot_true_vs_pred(np.concatenate([y_train, y_val]), np.concatenate([y_train_pred, y_val_pred]), "Train_Valid")
    plot_true_vs_pred(y_test, y_test_pred, "Test")
    analyze_branch_contribution(model, X_test_t, y_test_t)

