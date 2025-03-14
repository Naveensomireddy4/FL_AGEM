# Federated Learning with AGEM & FedAvg

This repository contains the implementation of Federated Learning (FL) using **Federated Averaging (FedAvg)** and **Average Gradient Episodic Memory (AGEM)**. The implementation includes both **client-side** and **server-side** functionalities.

## 📁 **Project Structure**
- **AGEM Code**: Located at `system/avalanche/training/plugins/agem.py`
- **Client-Side Code**: Located at `system/flcore/clients`
- **Server-Side Code**: Located at `system/flcore/servers`
- **Base Classes**:
  - `clientbase.py`: Base implementation for client-side FL.
  - `serverbase.py`: Base implementation for server-side FL.
  
## 🚀 **Running the Code**
To execute the training process, follow these steps:

### **1. Navigate to the System Directory**
```sh
cd system
```

### **2. Run the Training Script**
```sh
python3 main.py -data BloodMNIST -m resnet18 -algo FedAvg -gr 50 -did 0 -nc 3
```

## ⚙ **Command Parameters**
| Parameter | Description |
|-----------|-------------|
| `-data BloodMNIST` | Uses **BloodMNIST** dataset from MedMNIST. |
| `-m resnet18` | Uses **ResNet-18** as the model architecture. |
| `-algo FedAvg` | Runs the **Federated Averaging (FedAvg)** algorithm. |
| `-gr 50` | Sets **50** global communication rounds. |
| `-did 0` | Device ID (**0 for first available GPU**). Use `-1` for CPU. |
| `-nc 3` | Number of classes (ensure it matches the dataset's class count). |


