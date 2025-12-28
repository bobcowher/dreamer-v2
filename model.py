from numpy import int32
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class BaseModel(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_the_model(self, filename='models/latest.pt'):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)
        print(f"Saved model to {filename}")


    def load_the_model(self, filename='models/latest.pt', device='cuda'):
        try:
            self.load_state_dict(torch.load(filename, map_location=device))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")


class Encoder(BaseModel):

    def __init__(self, observation_shape=()):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)         # (1, C_enc, H_enc, W_enc)
            self.conv_output_shape = feats.shape[1:]   # (C_enc, H_enc, W_enc)
            self.flattened_dim = feats.numel() // 1    # C_enc * H_enc * W_enc
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        # conv_output = self._conv_forward(torch.zeros(1, *observation_shape))
        # conv_output_dim = conv_output.shape[-1]

        embed_dim = 64  # or whatever you pick

        # self.fc_enc = nn.Linear(self.flattened_dim, latent_dim)
        # self.fc_dec = nn.Linear(latent_dim, self.flattened_dim)

        self.fc_enc = nn.Linear(self.flattened_dim, embed_dim)  # ADD THIS
        self.fc_dec = nn.Linear(embed_dim, self.flattened_dim)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, observation_shape[0], kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d()

        print(f"VAE network initialized. Input shape: {observation_shape}")


    def _conv_features(self, x):
        # Convert uint8 to float if needed (for initialization)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x  # (B, C_enc, H_enc, W_enc)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x

    def _deconv_forward(self, x):
        x = x.view(-1, *self.conv_output_shape)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
       
        return x
        
    def encode(self, x):
        # x: (B,3,H,W) in [0,1]
        x = self._conv_forward(x)
        return x
    
    def decode(self, x):
        x = self._deconv_forward(x)
        return x

    def forward(self, x):
        # print(f"Type coming in: {type(x)}")
        embed = self.encode(x)
        # print(f"Embed type: {type(embed)}")
        recon = self.decode(embed)
        # print(f"Recon type: {type(recon)}")
        return recon, embed
