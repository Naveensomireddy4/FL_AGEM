import os
import copy
import time
import torch
import numpy as np
from flcore.clients.clientbase import Client
from utils.privacy import *
from avalanche.training.plugins.agem import AGEMPlugin  # Import AGEMPlugin
from torch import Tensor

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        Initializes the client with necessary configurations.
        Includes AGEMPlugin for episodic memory management and Differential Privacy (if enabled).
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Initialize AGEMPlugin with parameters
        self.agem_plugin = AGEMPlugin(patterns_per_experience=100, sample_size=1000)

        # Define client-specific episodic memory file path
        self.memory_filepath = f"episodic_memory_client_{self.id}.pkl"

        # Load episodic memory
        self.load_episodic_memory()

    def load_episodic_memory(self):
        """Load episodic memory from the file if it exists."""
        if os.path.exists(self.memory_filepath):
            self.agem_plugin.load_memory(self.memory_filepath)
            print(f"[Client {self.id}] Episodic memory loaded from {self.memory_filepath}.")
        else:
            print(f"[Client {self.id}] No episodic memory found, starting fresh.")

    def save_episodic_memory(self):
        """Save episodic memory to a file."""
        self.agem_plugin.save_memory(memory_path=self.memory_filepath)
        print(f"[Client {self.id}] Episodic memory saved to {self.memory_filepath}.")
    def train(self):
        """Performs local training with optional Differential Privacy and AGEM."""
        print(f"[Client {self.id}] Starting training...")

        # Ensure episodic memory is loaded
        self.load_episodic_memory()

        trainloader = self.load_train_data()
        self.model.train()

        # Initialize Differential Privacy if enabled
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
            print(f"[Client {self.id}] Initialized Differential Privacy.")

        start_time = time.time()

        # Adjust local epochs for slow training clientsccc
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            print(f"[Client {self.id}] Adjusted local epochs due to slow training: {max_local_epochs}")

        for epoch in range(max_local_epochs):
            print(f"[Client {self.id}] Epoch {epoch + 1}/{max_local_epochs}")
            for i, (x, y) in enumerate(trainloader):
                # Move data to the appropriate device
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # Simulate slow training if enabled
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Apply AGEM before the backward pass
                print(f"[Client {self.id}] Applying AGEM before training iteration...")
                self.agem_plugin.before_training_iteration(self)

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)
                print(f"[Client {self.id}] Loss: {loss.item()}")

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Apply AGEM after the backward pass
                print(f"[Client {self.id}] Applying AGEM after backward pass...")
                self.agem_plugin.after_backward(self)

                # Update model weights
                self.optimizer.step()
                print(f"[Client {self.id}] Updated weights.")

            # Update episodic memory after each epoch
            print(f"[Client {self.id}] Updating episodic memory after epoch {epoch + 1}...")
            self.update_memory_after_experience(dataset=trainloader.dataset)

        # Save episodic memory after training
        self.save_episodic_memory()

        # Apply learning rate decay if enabled
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            print(f"[Client {self.id}] Decayed learning rate.")

        # Update training time costs
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # Finalize Differential Privacy if enabled
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"[Client {self.id}] Differential Privacy stats: epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        print(f"[Client {self.id}] Training completed.")


    def update_memory_after_experience(self, dataset):
        """Update episodic memory after the experience."""
        print(f"[Client {self.id}] Updating episodic memory after experience...")
        self.agem_plugin.after_training_exp(self, dataset=dataset)

    def get_train_accuracy(self):
        """Calculate training accuracy."""
        print(f"[Client {self.id}] Calculating training accuracy...")

        # Ensure episodic memory is loaded
        self.load_episodic_memory()

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
        print(f"[Client {self.id}] Training accuracy: {accuracy}%")
        return accuracy

    def get_weights(self):
        """Retrieve model weights."""
        print(f"[Client {self.id}] Retrieving model weights.")
        return self.model.state_dict()

    def set_weights(self, weights):
        """Set model weights."""
        print(f"[Client {self.id}] Setting model weights.")
        self.model.load_state_dict(weights)
