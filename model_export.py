import torch
import torch.nn as nn
import numpy as np
import json
import os

# Define a simple GRU model
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # Only use the output from the last time step

# Instantiate and train your model (training code not included)
model = SimpleGRU(input_size=15, hidden_size=64, output_size=4, num_layers=3)



# After training, save the model weights and metadata
def save_model_weights(model, bin_filename, metadata_filename):
    # Get the state_dict
    state_dict = model.state_dict()

    # Convert weights to bytes and concatenate
    weights_bytes = b''.join(param.data.cpu().numpy().tobytes() for param in state_dict.values())

    # Prepare metadata
    metadata = {
        'model_name': model.__class__.__name__,
        'model_architecture': str(model),
        'num_params': sum(param.numel() for param in state_dict.values())  # Total number of parameters
    }

    for key, value in state_dict.items():
        metadata[key] = value.size()

    # Save weights and metadata
    with open(bin_filename, 'wb') as f:
        # Write weights
        f.write(weights_bytes)

    with open(metadata_filename, 'w') as f:
        # Write metadata
        json.dump(metadata, f)

# Example usage
save_model_weights(model, 'gru_weights.bin', 'gru_metadata.json')

# Verify saved file structure
print("Model weights and metadata saved successfully.")
