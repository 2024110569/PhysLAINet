import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Times New Roman']
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("vs", exist_ok=True)

BATCH_SIZE = 16
LR_PRE = 0.1
LR_FINE = 0.01
EPOCH_PRE = 100
EPOCH_FINE = 100
DROPOUT = 0.2

feature_cols = ["blue","green","red","rededge","nir","NDVI","NDRE","RVI","DVI","EVI","OSAVI"]
target_col = "LAI"

def load_numbers(file_path):
    with open(file_path,"r") as f:
        return list(map(int,f.read().strip().split(",")))

data = pd.read_excel("../data/params.xlsx")
data["number"] = data["number"].astype(int)
train_nums = load_numbers("../data/dataset_idx/train_numbers.txt")
val_nums = load_numbers("../data/dataset_idx/valid_numbers.txt")
test_nums = load_numbers("../data/dataset_idx/test_numbers.txt")

train_data = data[data["number"].isin(train_nums)]
val_data = data[data["number"].isin(val_nums)]
test_data = data[data["number"].isin(test_nums)]

X_train = train_data[feature_cols].values
y_train = train_data[target_col].values
X_val = val_data[feature_cols].values
y_val = val_data[target_col].values
X_test = test_data[feature_cols].values
y_test = test_data[target_col].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

def to_torch(x,y):
    return torch.tensor(x,dtype=torch.float32).to(DEVICE), torch.tensor(y,dtype=torch.float32).to(DEVICE).reshape(-1,1)

X_train_t,y_train_t = to_torch(X_train_s,y_train)
X_val_t,y_val_t = to_torch(X_val_s,y_val)
X_test_t,y_test_t = to_torch(X_test_s,y_test)

# ====================== 模型 ======================
class PROSAIL_DNN(nn.Module):
    def __init__(self,in_dim=11):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_dim,512),nn.Softplus(),nn.BatchNorm1d(512),nn.Dropout(DROPOUT),
            nn.Linear(512,256),nn.Softplus(),nn.BatchNorm1d(256),nn.Dropout(DROPOUT),
            nn.Linear(256,128),nn.Softplus(),nn.BatchNorm1d(128),nn.Dropout(DROPOUT),
            nn.Linear(128,64),nn.Softplus(),nn.BatchNorm1d(64),nn.Dropout(DROPOUT),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.model(x)

# ====================== 训练函数 ======================
def train(model,Xt,yt,Xv,yv,epochs,lr,save_path):
    criterion=nn.MSELoss()
    opt=optim.Adam(model.parameters(),lr=lr)
    sch=optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    best_r2=-999
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss=criterion(model(Xt),yt)
        loss.backward()
        opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_r2=r2_score(yv.cpu().numpy(),model(Xv).cpu().numpy())

        if val_r2>best_r2:
            best_r2=val_r2
            torch.save(model.state_dict(),save_path)
        if (epoch+1)%10==0:
            print(f"Epoch {epoch+1:2d} | Val R²: {val_r2:.4f}")
    print(f"Best: {save_path} | R²={best_r2:.4f}")

model=PROSAIL_DNN(in_dim=len(feature_cols)).to(DEVICE)
sim=pd.read_csv("prosail_simulation_9945.csv")
X_sim=sim[feature_cols].values
y_sim=sim[target_col].values
X_sim_s=scaler.transform(X_sim)

X_s_train,X_s_val,y_s_train,y_s_val=train_test_split(X_sim_s,y_sim,test_size=0.3,random_state=42)
X_s_train_t,y_s_train_t=to_torch(X_s_train,y_s_train)
X_s_val_t,y_s_val_t=to_torch(X_s_val,y_s_val)

train(model,X_s_train_t,y_s_train_t,X_s_val_t,y_s_val_t,EPOCH_PRE,LR_PRE,"checkpoints/best_pretrain.pth")

model.load_state_dict(torch.load("checkpoints/best_pretrain.pth",map_location=DEVICE))
train(model,X_train_t,y_train_t,X_val_t,y_val_t,EPOCH_FINE,LR_FINE,"checkpoints/best_finetune.pth")

def evaluate(model,X,y):
    model.eval()
    with torch.no_grad():
        pred=model(X).cpu().numpy().squeeze()
    true=y.cpu().numpy().squeeze()
    r2=r2_score(true,pred)
    rmse=np.sqrt(mean_squared_error(true,pred))
    rpd=np.std(true)/rmse if rmse>1e-6 else 0
    return true,pred,r2,rmse,rpd

model.load_state_dict(torch.load("checkpoints/best_finetune.pth",map_location=DEVICE))
t_test,p_test,r2_test,rmse_test,rpd_test=evaluate(model,X_test_t,y_test_t)
t_train,p_train,_,_,_=evaluate(model,X_train_t,y_train_t)
t_val,p_val,_,_,_=evaluate(model,X_val_t,y_val_t)

t_trainval = np.concatenate([t_train, t_val])
p_trainval = np.concatenate([p_train, p_val])

print("\n"+"="*50)
print(f"Results: ")
print(f"R² = {r2_test:.4f} | RMSE = {rmse_test:.4f} | RPD = {rpd_test:.4f}")
print("="*50)

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
    plt.savefig(f"vs/DNN_{target_col}_{dataset_name}.png", dpi=300)
    plt.show()

plot_true_vs_pred(t_train, p_train, "Train")
plot_true_vs_pred(t_val, p_val, "Valid")
plot_true_vs_pred(t_trainval, p_trainval, "Train_Valid")
plot_true_vs_pred(t_test, p_test, "Test")
model_type = "prosail_dnn"
os.makedirs(f"{model_type}_csv", exist_ok=True)
df_tv = pd.DataFrame({
    "LAI_true": t_trainval,
    "LAI_pred": p_trainval.flatten()
})
df_tv.to_csv(f"{model_type}_csv/{model_type}_tv.csv", index=False, encoding="utf-8-sig")
df_test = pd.DataFrame({
    "LAI_true": t_test,
    "LAI_pred": p_test.flatten()
})
df_test.to_csv(f"{model_type}_csv/{model_type}_t2025.csv", index=False, encoding="utf-8-sig")


joblib.dump(scaler,"checkpoints/scaler.pkl")
print("\n Done.")