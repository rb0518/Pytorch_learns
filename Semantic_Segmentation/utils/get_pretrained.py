import torch
from torchvision import models

print("start...")
model = models.resnet34(pretrained=True)
model.eval()
var=torch.ones(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resetnet34.pt")
print("end...")