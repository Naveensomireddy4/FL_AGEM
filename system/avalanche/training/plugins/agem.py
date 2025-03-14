
import os
import numpy as np
import logging
import gzip
import math
import random
import torch
import pickle
import zipfile
import warnings
from typing import List, Optional, Any, Iterator
from torch.utils.data import Dataset, Subset
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class AGEMPlugin(SupervisedPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """
        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.sample_size = int(sample_size)
        self.round_per_task = 10
        self.classes_per_task = 0

        # One AvalancheDataset for each experience
        self.buffers: List[AvalancheDataset] = []
        self.buffer_dataloader: Optional[GroupBalancedInfiniteDataLoader] = None
        # Placeholder iterator to avoid typing issues
        self.buffer_dliter: Iterator[Any] = iter([])
        # Placeholder Tensor to avoid typing issues
        self.reference_gradients: torch.Tensor = torch.empty(0)

    def load_memory(self, filepath: str):
        """Load episodic memory from the specified file."""
        try:
          #  print(f"[DEBUG] Loading memory from {filepath}...")

            if not os.path.exists(filepath):
                print(f"[Error] Memory file does not exist at {filepath}")
                return

            if os.path.getsize(filepath) == 0:
                print(f"[Error] Episodic memory file is empty: {filepath}")
                return

            # Check if the file has a '.npz' extension
            if not filepath.endswith('.npz'):
                print(f"[Error] The file {filepath} is not a valid .npz file.")
                return

            # Load the .npz file
            memory_data = np.load(filepath, allow_pickle=True)

            if isinstance(memory_data, np.lib.npyio.NpzFile):
                if 'buffers' in memory_data.files:
                    self.buffers = memory_data['buffers'].tolist()  # Convert to list
                   # print(f"[DEBUG] Episodic memory loaded successfully from {filepath}")
                     # Dataloader configuration
                    def custom_collate_fn(batch):
                        inputs, labels = zip(*batch)
                        return torch.stack(inputs), torch.stack(labels)
                    num_workers = 0
                    persistent_workers = num_workers > 0
                    self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
                        self.buffers,
                        batch_size=(self.sample_size // len(self.buffers)),
                        num_workers=num_workers,
                        pin_memory=False,
                        persistent_workers=persistent_workers,
                        collate_fn=custom_collate_fn
                     )
                    self.buffer_dliter = iter(self.buffer_dataloader)

                else:
                    print(f"[Error] 'buffers' key not found in memory file: {filepath}")
            else:
                print(f"[Error] Unexpected file format in {filepath}")

        except zipfile.BadZipFile:
            print(f"[Error] The file {filepath} is not a valid zip file or is corrupted.")
        except Exception as e:
            print(f"[Error] Failed to load episodic memory: {e}")

    def before_training_iteration(self, strategy , round, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        #print("[DEBUG] before_training_iteration: Starting training iteration...")

        if len(self.buffers) > 0:
          #  print("WE ARE DOING BEFORE TRAINING ITERATION")
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()

            # Check if minibatch is valid
            if mb is None or len(mb) == 0:
              #  print(f"[Error] No valid data in minibatch, skipping...")
                return

            try:
                xref, yref, tid = mb[0], mb[1], mb[-1]
                xref, yref = xref.to(strategy.device), yref.to(strategy.device)
               # print("[DEBUG] Unpacked minibatch successfully.")
            except IndexError:
                print(f"[Error] Error unpacking minibatch. Skipping...")
                return  # Skip if unpacking fails

            import math

            # Forward pass using Avalanche
            out = avalanche_forward(strategy.model, xref, tid, task_id=math.floor(round / self.round_per_task))

            # Print output shape and min/max values
            # print(f"Output shape: {out.shape}")
            # print(f"Output min: {out.min().item()}, Output max: {out.max().item()}")

            # Compute the task index
            round_floor = math.floor(round / self.round_per_task)

            # Print yref min/max before adjustment
            # print(f"yref before adjustment - min: {yref.min().item()}, max: {yref.max().item()}")
            # print(f"Unique labels in yref: {yref.unique().tolist()}")


            # Adjust labels
            if(yref.min() > 0 ):
                yref -= round_floor * self.classes_per_task

            # Print yref min/max after adjustment
            # print(f"yref after adjustment - min: {yref.min().item()}, max: {yref.max().item()}")
            # print(f"Unique labels in yref: {yref.unique().tolist()}")


            # Ensure label values are in a valid range
            if (yref < 0).any():
                raise ValueError("Adjusted yref contains negative values. Check label transformation.")

            # Compute loss
            loss = strategy.loss(out, yref)

            # Backpropagation
            loss.backward()


            # Gradient projection for multi-headed models where gradient can be None
            reference_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(reference_gradients_list)
            strategy.optimizer.zero_grad()

           # print("[DEBUG] Reference gradient computed and saved.")

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients.
        """
      #  print("[DEBUG] after_backward: Projecting gradients...")

        if len(self.buffers) > 0:
            current_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients, self.reference_gradients)
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param

             #   print("[DEBUG] Gradients projected and applied to model.")

    def after_training_exp(self, strategy, dataset,round=0, **kwargs):
        """Update replay memory with patterns from current experience."""
      #  print("[DEBUG] after_training_exp: Updating memory with current experience...")
        self.update_memory(dataset, **kwargs,round=round)

    def sample_from_memory(self):
        """Sample from the memory buffer."""
       # print("[DEBUG] sample_from_memory: Sampling from memory...")

        if self.buffer_dliter is None:
            print("[Error] Buffer data loader is not initialized.")
            return None

        if len(self.buffers) == 0:
            print("[Error] No buffers available for sampling.")
            return None

        if self.buffer_dataloader is None:
            print("[Error] Buffer data loader is None, cannot sample.")
            return None

        try:
            sample = next(self.buffer_dliter)
          #  print("[DEBUG] Sample retrieved from memory.")
            return sample
        except StopIteration:
            print("[Warning] Buffer data loader is empty or exhausted. Reinitializing...")

            # If exhausted, reinitialize the buffer data loader
            self.buffer_dliter = iter(self.buffer_dataloader)

            try:
                sample = next(self.buffer_dliter)
             #   print("[DEBUG] Sample retrieved after reinitialization.")
                return sample
            except StopIteration:
                print("[Error] Unable to sample from the buffer after reinitialization.")
                return None

    def update_memory(self, dataset, num_workers: int = 0, max_buffers: int = 5,round=0, **kwargs):
        """Update replay memory with patterns from current experience, maintaining a fixed size."""
       # print(f"[DEBUG] update_memory: Updating memory with dataset of size {len(dataset)}...")

        if len(dataset) == 0:
            print("[Error] Dataset is empty during memory update.")
            return

       # print(f"[Debug] Updating memory with {len(dataset)} samples.")
        if num_workers > 0:
            warnings.warn("Num workers > 0 is known to cause heavy slowdowns in AGEM.")

        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset)

        # Limit dataset size to patterns_per_experience
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
           # print(f"dataset size exceed . Storing only {self.patterns_per_experience}")
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = Subset(dataset, indices[:self.patterns_per_experience])

        # Append new dataset and maintain a fixed size for buffers
        x= len(self.buffers)
        #print(f"*********dataset length is {len(dataset)}***********")
        if(round>0 and (round+1)%self.round_per_task == 0):
            self.buffers.append(dataset)
        
            
        print(f"*********New episodic memory length is {len(self.buffers)}***********")
        
        if len(self.buffers) > max_buffers:
            removed_dataset = self.buffers.pop(0)
         #   print(f"[Info] Removed oldest dataset to maintain buffer size. Current size: {len(self.buffers)}")

        # Dataloader configuration
        def custom_collate_fn(batch):
            inputs, labels = zip(*batch)
            return torch.stack(inputs), torch.stack(labels)

        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=(self.sample_size // len(self.buffers)),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers,
            collate_fn=custom_collate_fn
        )
        self.buffer_dliter = iter(self.buffer_dataloader)

    def save_memory(self, memory_path, compress: bool = True, use_npz: bool = True, reduce_precision: bool = False):
        """
        Save episodic memory to a file. Supports saving in .npz format (with or without compression)
        or using pickle and gzip, with optional reduced precision for tensors.
        """
       # print(f"[DEBUG] save_memory: Saving memory to {memory_path}...")

        try:
            if reduce_precision:
                print("[Info] Reducing precision of stored memory data to float16.")
                self.buffers = [
                    [data.half() if isinstance(data, torch.Tensor) else data for data in experience]
                    for experience in self.buffers
                ]

            if use_npz:
                buffers_array = np.array(self.buffers, dtype=object)
                if compress:
                    np.savez_compressed(memory_path, buffers=buffers_array)
                else:
                    np.savez(memory_path, buffers=buffers_array)
              #  print(f"[Success] Episodic memory saved to {memory_path} as a .npz file.")
            else:
                with open(memory_path, 'wb') as f:
                    pickle.dump(self.buffers, f)
               # print(f"[Success] Episodic memory saved to {memory_path} using pickle.")

        except Exception as e:
            print(f"[Error] Failed to save episodic memory: {e}")





class DatasetFromList(Dataset):
    """Helper class to wrap a list of data into a Dataset."""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
