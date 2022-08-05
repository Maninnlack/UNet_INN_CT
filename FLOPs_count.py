import torch
from thop import profile
from torchstat import stat

from Unet_ANN import UNet
from Unet_INN import UNet_INN

input = torch.randn(1, 3, 256, 256)

model_i = UNet_INN(3, 1)
model_a = UNet(3, 1)

# thop counting
# macs_u, params_u = profile(model_u, inputs=(input, ))
# macs_a, params_a = profile(model_a, inputs=(input, ))

# print(f"UNet_INN mac: {macs_u} param: {params_u}")
# print(f"UNet_ANN mac: {macs_a} param: {params_a}")

# torchstat counting
stat(model_a, (3, 256, 256))



