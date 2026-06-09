import torch

def count_params_from_pth(pth_path):
    state_dict = torch.load(pth_path, map_location="cpu")
    total = 0
    for key, param in state_dict.items():
        if "optimizer" not in key and "epoch" not in key and "step" not in key:
            if isinstance(param, torch.Tensor):
                total += param.numel()
    print(f"params: {total:,}")
    print(f"total: {total / 1e6:.2f} M")
    return total
count_params_from_pth(r"physlainet.pth")