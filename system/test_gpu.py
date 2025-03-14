import torch
from flcore.trainmodel.my import *
import torch

# Ensure you are using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to the same device
model = ResNet18_MultiHead(in_channels=3, task_classes={0: 10, 1: 5})
model.to(device)  # Move model to GPU

# Set task before inference
model.set_task(0)

# Create input tensor and move it to the same device
x = torch.randn(1, 3, 224, 224).to(device)

# Forward pass
output = model(x)
print(output.shape)  # Should match the number of classes for task 0
