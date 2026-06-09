import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import optuna
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
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
def objective(trial):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (64, 32), (128, 64)])
    activation = trial.suggest_categorical("activation", ["relu"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-2)
    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    return mse


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=50)
print("Best hyperparameters:", study.best_params)
print("Best validation MSE:", study.best_value)
X_trainval_scaled = scaler.fit_transform(X_train)
y_trainval = y_train
final_model = MLPRegressor(
    hidden_layer_sizes=study.best_params["hidden_layer_sizes"],
    activation=study.best_params["activation"],
    solver=study.best_params["solver"],
    alpha=study.best_params["alpha"],
    learning_rate_init=study.best_params["learning_rate_init"],
    max_iter=1000,
    random_state=RANDOM_SEED
)
final_model.fit(X_trainval_scaled, y_trainval)
X_test_scaled_final = scaler.transform(X_test)
y_test_pred = final_model.predict(X_test_scaled_final)
def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    return r2, rmse
y_trainval = np.concatenate([y_train, y_val])
X_trainval_scaled = scaler.transform(np.concatenate([X_train, X_val]))
y_trainval_pred = final_model.predict(X_trainval_scaled)
evaluate_model(y_trainval, y_trainval_pred, "Training + Validation Set")
evaluate_model(y_test, y_test_pred, "Test Set")
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
    plt.savefig(f"vs/NN_{target_col}_{dataset_name}.png", dpi=300)
    plt.show()
plot_true_vs_pred(y_trainval, y_trainval_pred, "Train_Valid")
plot_true_vs_pred(y_test, y_test_pred, "Test")
joblib.dump(final_model, f"checkpoints/mlp_{target_col}_model.pkl")
joblib.dump(scaler, f"checkpoints/mlp_{target_col}_scaler.pkl")