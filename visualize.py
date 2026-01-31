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
    Compare reconstructions to diagnose where information is lost:
    1. Original image
    2. Posterior features (h, z from RSSM with observation)
    3. Prior features (h, z_prior from RSSM without observation - imagination)
    
    This reveals if the posterior is actually using observation info.
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    
    with torch.no_grad():
        # Encode observations
        embed = world_model.encoder(obs_flat)  # (N, 1024)
        embeds = embed.view(num_samples, 1, -1)
        
        # === PATH 1: Posterior (uses observation) ===
        h_all, z_all, prior_logits, post_logits = world_model.rssm.observe_sequence(actions, embeds)
        post_features = torch.cat([h_all, z_all], dim=-1)
        post_features_flat = post_features.view(num_samples, -1)
        post_recon = world_model.decoder(post_features_flat)
        
        # === PATH 2: Prior only (no observation - what imagination sees) ===
        # Sample z from prior instead of posterior
        z_prior = world_model.rssm.sample_stochastic(prior_logits.view(num_samples, -1))
        prior_features = torch.cat([h_all.view(num_samples, -1), z_prior], dim=-1)
        prior_recon = world_model.decoder(prior_features)
        
        # === Measure posterior vs prior divergence ===
        post_probs = F.softmax(post_logits.view(-1, 32, 32), dim=-1)
        prior_probs = F.softmax(prior_logits.view(-1, 32, 32), dim=-1)
        kl_per_var = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(dim=-1)
        total_kl = kl_per_var.sum(dim=-1).mean()  # Total KL averaged over batch
    
    # Convert to numpy HWC format
    originals = obs_flat.permute(0, 2, 3, 1).cpu().numpy()
    post_recons = (post_recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    prior_recons = (prior_recon.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Add labels
    label_height = 20
    H, W = originals.shape[1], originals.shape[2]
    
    def add_label(img, text):
        label_bar = np.zeros((label_height, W, 3), dtype=np.uint8)
        cv2.putText(label_bar, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return np.vstack([label_bar, img])
    
    # Build grid: each column is [original, posterior, prior] for one sample
    columns = []
    for i in range(num_samples):
        orig_labeled = add_label(originals[i], f"Original {i+1}")
        post_labeled = add_label(post_recons[i], f"Posterior {i+1}")
        prior_labeled = add_label(prior_recons[i], f"Prior {i+1}")
        column = np.vstack([orig_labeled, post_labeled, prior_labeled])
        columns.append(column)
    
    grid = np.hstack(columns)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, grid_bgr)
    print(f"Saved {filename}")
    
    # Print diagnostics
    obs_norm = obs_flat.float() / 255.0
    post_l1 = F.l1_loss(post_recon, obs_norm).item()
    prior_l1 = F.l1_loss(prior_recon, obs_norm).item()
    print(f"  Posterior L1: {post_l1:.4f}")
    print(f"  Prior L1:     {prior_l1:.4f}")
    print(f"  KL(post||prior): {total_kl:.4f} nats")
    print(f"  If posterior ≈ prior images, the observation isn't being used!")


if __name__ == "__main__":
    print("Import this module and call:")
    print("  visualize_reconstruction(world_model, memory)")
    print("  visualize_bypass_test(world_model, memory)")
