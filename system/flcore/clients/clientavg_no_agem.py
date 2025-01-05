import copy
import torch
import numpy as np
import time
import wandb
from flcore.clients.clientbase import Client
from utils.privacy import *

# Initialize W&B at the beginning of your script
wandb.init(project='fl', entity='iiest')

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # Track initial weights
        wandb.watch(self.model, log='all')

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Log training loss for each epoch
            wandb.log({'client_id': self.id, 'epoch': epoch, 'loss': epoch_loss / len(trainloader)})

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            # Log differential privacy metrics
            wandb.log({'client_id': self.id, 'epsilon': eps, 'delta': DELTA})

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def get_train_accuracy(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.load_train_data():
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = inputs[0].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.model.train()  # Switch back to training mode
        accuracy = 100 * correct / total

        # Log training accuracy
        wandb.log({'client_id': self.id, 'train_accuracy': accuracy})

        return accuracy

    def get_weights(self):
        return self.model.state_dict()  
    
    def set_weights(self, weights):
        self.model.load_state_dict(weights)
