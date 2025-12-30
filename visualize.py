import torch
import numpy as np
import cv2


def visualize_reconstruction(world_model, memory, num_samples=4, filename="recon_check.png"):
    """
    Save a grid comparing original images (top row) vs reconstructions (bottom row).
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    actions_flat = actions.view(num_samples, *actions.shape[2:])
    
    with torch.no_grad():
        # Encode observations
        embed = world_model.encoder(obs_flat)
        embeds = embed.view(num_samples, 1, -1)
        
        # Get states from RSSM
        h_all, z_all, _, _ = world_model.rssm.observe_sequence(actions, embeds)
        
        # Decode from features
        features = torch.cat([h_all, z_all], dim=-1)
        features_flat = features.view(num_samples, -1)
        recon = world_model.decoder(features_flat)
    
    # Convert to numpy HWC format
    originals = obs_flat.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    recons = (recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Add labels
    label_height = 20
    H, W = originals.shape[1], originals.shape[2]
    
    def add_label(img, text):
        """Add a label bar above the image."""
        label_bar = np.zeros((label_height, W, 3), dtype=np.uint8)
        cv2.putText(label_bar, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return np.vstack([label_bar, img])
    
    # Build grid: each column is [original, reconstruction] for one sample
    columns = []
    for i in range(num_samples):
        orig_labeled = add_label(originals[i], f"Original {i+1}")
        recon_labeled = add_label(recons[i], f"Recon {i+1}")
        column = np.vstack([orig_labeled, recon_labeled])
        columns.append(column)
    
    # Stack columns horizontally
    grid = np.hstack(columns)
    
    # Convert RGB to BGR for cv2
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, grid_bgr)
    print(f"Saved {filename}")


if __name__ == "__main__":
    # Quick test
    print("Import this module and call visualize_reconstruction(encoder, memory)")
