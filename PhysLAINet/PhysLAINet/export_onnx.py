import torch
from cons_chain import PhysLAINet

model = PhysLAINet(input_dim=30)
model.load_state_dict(torch.load("checkpoints/physlainet.pth", map_location='cpu'))
model.eval()
dummy_input = torch.randn(1, 30)
torch.onnx.export(model, dummy_input, "checkpoints/physlainet.onnx", input_names=['input'], output_names=['output'], opset_version=13)