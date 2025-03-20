import os
import copy
import time
import random
import torch
import math
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from avalanche.training.plugins.agem import AGEMPlugin
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from typing import Any, List, Tuple, Optional
from flcore.clients.clientbase import Client
from utils.privacy import initialize_dp, get_dp_params


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        Initializes the client with necessary configurations.
        Includes AGEMPlugin for episodic memory management and Differential Privacy (if enabled).
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Initialize AGEMPlugin
        self.round_per_task = 2
        self.classes_per_task = 0

        self.agem_plugin = AGEMPlugin(patterns_per_experience=30, sample_size=30)

        # Episodic memory file path
        self.memory_filepath = f"episodic_memory_client_{self.id}.npz"

        # Load episodic memory
        #self.load_episodic_memory()

    def load_episodic_memory(self):
        """Load episodic memory from the file if it exists."""
        if os.path.exists(self.memory_filepath):
            self.agem_plugin.load_memory(self.memory_filepath)
          #  print(f"[Client {self.id}] Episodic memory loaded from {self.memory_filepath}.")
        else:
            print(f"[Client {self.id}] No episodic memory found, starting fresh.")

    def save_episodic_memory(self):
        """Save episodic memory to a file."""
        print(f"file path {self.memory_filepath}")
        self.agem_plugin.save_memory(memory_path=self.memory_filepath)
       # print(f"[Client {self.id}] Episodic memory saved to {self.memory_filepath}.")

    def train(self,round):
        """Perform local training with optional Differential Privacy and AGEM."""
      #  print(f"[Client {self.id}] Starting training...")

        # Ensure episodic memory is loaded
        self.load_episodic_memory()

        trainloader = self.load_train_data(round = math.floor(round/self.round_per_task ))
        self.model.train()

        # Initialize Differential Privacy if enabled
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )
            print(f"[Client {self.id}] Initialized Differential Privacy.")

        start_time = time.time()

        # Adjust local epochs for slow training clients
        max_local_epochs = self.local_epochs if not self.train_slow else np.random.randint(1, self.local_epochs // 2)
        #print(f"[Client {self.id}] Training for {max_local_epochs} epochs.")

        for epoch in range(max_local_epochs):
            #print(f"[Client {self.id}] Epoch {epoch + 1}/{max_local_epochs}")
            for i, (x, y) in enumerate(trainloader):
                # Move data to the appropriate device
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                

                # Simulate slow training if enabled
                if self.train_slow:
                    time.sleep(0.1 * np.random.rand())

                # Apply AGEM before the backward pass
                if(round>(self.round_per_task-1)):
                    self.agem_plugin.before_training_iteration(self,round = round)

                # Forward pass
                task_id=math.floor(round / self.round_per_task )
                y = y - (task_id * 20)  


                output = self.model(x, task_id=math.floor(round / self.round_per_task ))
                # print(f"[Client {self.id}] y shape: {y.shape}, min: {y.min().item()}, max: {y.max().item()}, unique values: {y.unique()}")
                # print(f"[Client {self.id}] output shape: {output.shape}, expected classes: {output.shape[1]}")  

                loss = self.loss(output, y)

                #print(f"[Client {self.id}] Loss: {loss.item()}")

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Apply AGEM after the backward pass
                if(round>(self.round_per_task-1)):
                    self.agem_plugin.after_backward(self)

                # Update model weights
                self.optimizer.step()

            
       # Update episodic memory after each epoch
        if(round>=(self.round_per_task-1)):
            self.agem_plugin.after_training_exp(self, dataset=trainloader.dataset,round=round)
            # Save episodic memory
            self.save_episodic_memory()
        

        # Apply learning rate decay if enabled
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            print(f"[Client {self.id}] Learning rate decayed.")

        # Update training time costs
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # Finalize Differential Privacy if enabled
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"[Client {self.id}] Differential Privacy stats: epsilon = {eps:.2f}, sigma = {DELTA}")

            # Restore original model
            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

      #  print(f"[Client {self.id}] Training completed.")

    def update_memory_after_experience(self, dataset):
        """Update episodic memory after the experience."""
       # print(f"[Client {self.id}] Updating episodic memory after experience...")

        if len(dataset) > self.agem_plugin.patterns_per_experience:
            # Shuffle and subset
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = [dataset[i] for i in indices[:self.agem_plugin.patterns_per_experience]]

        # Convert to PyTorch tensors
        inputs, labels = zip(*dataset)
        inputs = torch.stack([torch.tensor(inp) for inp in inputs])
        labels = torch.tensor(labels)
        dataset = TensorDataset(inputs, labels)

        self.agem_plugin.buffers.append(dataset)

        # Define a custom collate function
        def custom_collate_fn(batch):
            inputs, labels = zip(*batch)
            return torch.stack(inputs), torch.stack(labels)

        self.agem_plugin.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.agem_plugin.buffers,
            batch_size=self.agem_plugin.sample_size // len(self.agem_plugin.buffers),
            num_workers=0,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )
        self.agem_plugin.buffer_dliter = iter(self.agem_plugin.buffer_dataloader)

    def get_train_accuracy(self):
        """Calculate training accuracy."""
      #  print(f"[Client {self.id}] Calculating training accuracy...")

        self.load_episodic_memory()
        correct, total = 0, 0
        self.model.eval()

        with torch.no_grad():
            for data in self.load_train_data():
                inputs, labels = data
                inputs = inputs[0].to(self.device) if isinstance(inputs, list) else inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()  # Switch back to training mode
        accuracy = 100 * correct / total
     #   print(f"[Client {self.id}] Training accuracy: {accuracy}%")
        return accuracy

    def get_weights(self):
        """Retrieve model weights."""
       # print(f"[Client {self.id}] Retrieving model weights.")
        return self.model.state_dict()

    def set_weights(self, weights):
        """Set model weights."""
      #  print(f"[Client {self.id}] Setting model weights.")
        self.model.load_state_dict(weights)
