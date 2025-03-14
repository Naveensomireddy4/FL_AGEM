import numpy as np
import matplotlib.pyplot as plt
import torch  # Import PyTorch to handle the tensor

# Load the .npz file with allow_pickle=True
data = np.load(r'C:\Users\navee\IIIT-DELHI\PFLlib\system\episodic_memory_client_1.npz', allow_pickle=True)

# List all keys
print("Keys in the .npz file:", data.files)

# Access and print the 'buffers' key
buffers = data['buffers']
print("Type of 'buffers':", type(buffers))
print("Shape of 'buffers':", buffers.shape)

# Inspect the first buffer's type and content
first_buffer = buffers[0]
print("Type of first buffer:", type(first_buffer))
print("Content of first buffer:", first_buffer)

# Extract and process the tensor
if isinstance(first_buffer, np.ndarray):
    # Check if the first element inside contains a tensor
    if isinstance(first_buffer[0][0], torch.Tensor):
        # Extract the first tensor (image) and label from the first buffer
        image_tensor = first_buffer[0][0].numpy()
        label_tensor = first_buffer[0][1]  # Assuming the second element is the label

        # Normalize or handle the data if needed (assuming values are in [0, 1])
        image = np.clip(image_tensor, 0, 1)  # Ensure values are within [0, 1] range

        # Convert label_tensor to an integer if it's a PyTorch tensor
        if isinstance(label_tensor, torch.Tensor):
            label = label_tensor.item()
        else:
            label = label_tensor

        # Display the image with its label
        plt.imshow(image)
        plt.title(f"Image Label: {label}")
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("First buffer does not contain a tensor.")
else:
    print("The first buffer is not an array of tensors and cannot be displayed as an image.")
