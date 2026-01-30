import torch
import torch.nn.functional as F
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


def visualize_bypass_test(world_model, memory, num_samples=4, filename="bypass_check.png"):
    """
    Compare three reconstructions:
    1. Original image
    2. Direct encoder→decoder (bypass RSSM) 
    3. Full RSSM path
    
    This reveals where information is being lost.
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    
    with torch.no_grad():
        # Encode observations
        embed = world_model.encoder(obs_flat)  # (N, 1024)
        
        # === PATH 1: Direct bypass (encoder → decoder, no RSSM) ===
        # Pad embed to match feature_dim (1024 → 1536)
        padded = F.pad(embed, (0, 512))  # (N, 1536)
        direct_recon = world_model.decoder(padded)
        
        # === PATH 2: Through RSSM ===
        embeds = embed.view(num_samples, 1, -1)
        h_all, z_all, _, _ = world_model.rssm.observe_sequence(actions, embeds)
        features = torch.cat([h_all, z_all], dim=-1)
        features_flat = features.view(num_samples, -1)
        rssm_recon = world_model.decoder(features_flat)
    
    # Convert to numpy HWC format
    originals = obs_flat.permute(0, 2, 3, 1).cpu().numpy()
    direct_recons = (direct_recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    rssm_recons = (rssm_recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Add labels
    label_height = 20
    H, W = originals.shape[1], originals.shape[2]
    
    def add_label(img, text):
        label_bar = np.zeros((label_height, W, 3), dtype=np.uint8)
        cv2.putText(label_bar, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return np.vstack([label_bar, img])
    
    # Build grid: each column is [original, direct, rssm] for one sample
    columns = []
    for i in range(num_samples):
        orig_labeled = add_label(originals[i], f"Original {i+1}")
        direct_labeled = add_label(direct_recons[i], f"Direct {i+1}")
        rssm_labeled = add_label(rssm_recons[i], f"RSSM {i+1}")
        column = np.vstack([orig_labeled, direct_labeled, rssm_labeled])
        columns.append(column)
    
    grid = np.hstack(columns)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, grid_bgr)
    print(f"Saved {filename}")
    
    # Also print L1 errors for quantitative comparison
    obs_norm = obs_flat.float() / 255.0
    direct_l1 = F.l1_loss(direct_recon, obs_norm).item()
    rssm_l1 = F.l1_loss(rssm_recon, obs_norm).item()
    print(f"  Direct (bypass) L1: {direct_l1:.4f}")
    print(f"  RSSM path L1:       {rssm_l1:.4f}")


if __name__ == "__main__":
    print("Import this module and call:")
    print("  visualize_reconstruction(world_model, memory)")
    print("  visualize_bypass_test(world_model, memory)")
