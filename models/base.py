import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    
    def __init__(self):
        super().__init__()

    def save_the_model(self, filename, verbose=False):
        # Create directory if it doesn't exist
        path = "checkpoints/" + filename + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

        if verbose:
            print(f"Saved model to {path}")

    def load_the_model(self, filename, device='cuda'):
    
        path = "checkpoints/" + filename + ".pt"

        try:
            self.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded weights from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

