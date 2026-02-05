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


def diagnose_encoder_embeddings(world_model, memory, num_samples=8):
    """
    Check if the encoder produces meaningfully different embeddings
    for different observations.
    
    If embeddings are all similar, the encoder is the bottleneck.
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    
    with torch.no_grad():
        embeds = world_model.encoder(obs_flat)  # (N, 1024)
        
        # Compute pairwise L2 distances between embeddings
        # Normalize first to make distances comparable
        embeds_norm = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Pairwise cosine similarity
        similarity_matrix = embeds_norm @ embeds_norm.T  # (N, N)
        
        # Get off-diagonal similarities (exclude self-similarity)
        mask = ~torch.eye(num_samples, dtype=torch.bool, device=similarity_matrix.device)
        off_diag_sims = similarity_matrix[mask]
        
        mean_sim = off_diag_sims.mean().item()
        min_sim = off_diag_sims.min().item()
        max_sim = off_diag_sims.max().item()
        
        # Also check embedding statistics
        embed_mean = embeds.mean().item()
        embed_std = embeds.std().item()
        embed_norm = embeds.norm(dim=-1).mean().item()
    
    print("=" * 50)
    print("ENCODER EMBEDDING ANALYSIS")
    print("=" * 50)
    print(f"  Embedding stats:")
    print(f"    Mean:     {embed_mean:.4f}")
    print(f"    Std:      {embed_std:.4f}")
    print(f"    Avg norm: {embed_norm:.4f}")
    print(f"  Pairwise cosine similarity (different observations):")
    print(f"    Mean: {mean_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")
    print()
    if mean_sim > 0.95:
        print("  ⚠️  Embeddings are very similar (>0.95 cosine sim)!")
        print("  Encoder may not be capturing observation differences.")
    elif mean_sim > 0.8:
        print("  ⚠️  Embeddings are fairly similar. Some differentiation.")
    else:
        print("  ✓ Embeddings show good differentiation between observations.")
    print("=" * 50)


def diagnose_posterior_outputs(world_model, memory, num_samples=8):
    """
    Check if the posterior produces meaningfully different z distributions
    for different observations.
    
    Even if encoder embeds are different, the posterior might collapse them.
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    
    with torch.no_grad():
        embed = world_model.encoder(obs_flat)
        embeds = embed.view(num_samples, 1, -1)
        
        h_all, z_all, prior_logits, post_logits = world_model.rssm.observe_sequence(actions, embeds)
        
        # Get posterior probabilities
        post_probs = F.softmax(post_logits.view(num_samples, 32, 32), dim=-1)  # (N, 32, 32)
        post_flat = post_probs.view(num_samples, -1)  # (N, 1024)
        
        # Pairwise cosine similarity of posterior distributions
        post_norm = post_flat / (post_flat.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_matrix = post_norm @ post_norm.T
        
        mask = ~torch.eye(num_samples, dtype=torch.bool, device=similarity_matrix.device)
        off_diag_sims = similarity_matrix[mask]
        
        mean_sim = off_diag_sims.mean().item()
        min_sim = off_diag_sims.min().item()
        max_sim = off_diag_sims.max().item()
        
        # Check entropy of posteriors (high entropy = uncertain/uniform)
        entropy = -(post_probs * torch.log(post_probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = np.log(32)  # Maximum entropy for 32 classes
        
    print("=" * 50)
    print("POSTERIOR DISTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"  Pairwise cosine similarity (different observations):")
    print(f"    Mean: {mean_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")
    print(f"  Posterior entropy: {entropy:.4f} (max: {max_entropy:.4f})")
    print()
    if mean_sim > 0.95:
        print("  ⚠️  Posterior distributions are nearly identical!")
        print("  The posterior is collapsing - not using observation info.")
    elif mean_sim > 0.8:
        print("  ⚠️  Posteriors are fairly similar. Weak differentiation.")
    else:
        print("  ✓ Posteriors show good differentiation between observations.")
    
    if entropy > 0.9 * max_entropy:
        print("  ⚠️  Posterior entropy is near maximum - distributions are too uniform.")
    elif entropy < 0.3 * max_entropy:
        print("  ✓ Posterior is confident (low entropy).")
    print("=" * 50)


def diagnose_decoder_weights(world_model):
    """
    Check if the decoder's first layer has learned to ignore z.
    
    The decoder input is [h, z] = [hidden_dim, stoch_flat] = [512, 1024] = 1536
    If weights for the z portion are near zero, decoder is ignoring z.
    """
    # Get first layer weights
    fc_weights = world_model.decoder.fc_dec.weight  # (out_features, in_features) = (6144, 1536)
    
    hidden_dim = world_model.rssm.hidden_dim  # 512
    stoch_flat = world_model.rssm.stoch_flat  # 1024
    
    # Split weights into h portion and z portion
    h_weights = fc_weights[:, :hidden_dim]      # (6144, 512)
    z_weights = fc_weights[:, hidden_dim:]      # (6144, 1024)
    
    # Compute statistics
    h_norm = h_weights.norm().item()
    z_norm = z_weights.norm().item()
    h_mean_abs = h_weights.abs().mean().item()
    z_mean_abs = z_weights.abs().mean().item()
    
    print("=" * 50)
    print("DECODER FIRST LAYER WEIGHT ANALYSIS")
    print("=" * 50)
    print(f"  h weights (first {hidden_dim} dims):")
    print(f"    L2 norm:  {h_norm:.4f}")
    print(f"    Mean|w|:  {h_mean_abs:.6f}")
    print(f"  z weights (last {stoch_flat} dims):")
    print(f"    L2 norm:  {z_norm:.4f}")
    print(f"    Mean|w|:  {z_mean_abs:.6f}")
    print(f"  Ratio (h/z norms): {h_norm/z_norm:.2f}")
    print()
    if h_norm / z_norm > 2:
        print("  ⚠️  h weights are much larger than z weights!")
        print("  The decoder may be ignoring z.")
    elif z_norm / h_norm > 2:
        print("  ✓ z weights are larger - decoder should use z.")
    else:
        print("  Weights are roughly balanced.")
    print("=" * 50)


def test_z_sensitivity(world_model, memory, num_samples=4):
    """
    Directly test decoder sensitivity to z by:
    1. Get real (h, z) features
    2. Decode with real z
    3. Decode with z = zeros
    4. Decode with z = random noise
    
    If outputs are similar, decoder ignores z.
    """
    obs, actions, _, _, _ = memory.sample_buffer(num_samples, sequence_length=1)
    obs_flat = obs.view(num_samples, *obs.shape[2:])
    
    with torch.no_grad():
        embed = world_model.encoder(obs_flat)
        embeds = embed.view(num_samples, 1, -1)
        
        h_all, z_all, _, _ = world_model.rssm.observe_sequence(actions, embeds)
        h = h_all.view(num_samples, -1)
        z = z_all.view(num_samples, -1)
        
        # Three conditions
        features_real = torch.cat([h, z], dim=-1)
        features_zero_z = torch.cat([h, torch.zeros_like(z)], dim=-1)
        features_rand_z = torch.cat([h, torch.randn_like(z)], dim=-1)
        
        recon_real = world_model.decoder(features_real)
        recon_zero_z = world_model.decoder(features_zero_z)
        recon_rand_z = world_model.decoder(features_rand_z)
        
        # Measure differences
        diff_zero = F.l1_loss(recon_real, recon_zero_z).item()
        diff_rand = F.l1_loss(recon_real, recon_rand_z).item()
        
    print("=" * 50)
    print("DECODER Z-SENSITIVITY TEST")
    print("=" * 50)
    print(f"  L1(real_z vs zero_z): {diff_zero:.6f}")
    print(f"  L1(real_z vs rand_z): {diff_rand:.6f}")
    print()
    if diff_zero < 0.01 and diff_rand < 0.01:
        print("  ⚠️  Decoder output barely changes when z changes!")
        print("  The decoder is ignoring z entirely.")
    elif diff_zero < 0.05:
        print("  ⚠️  Decoder has weak sensitivity to z.")
    else:
        print("  ✓ Decoder is sensitive to z values.")
    print("=" * 50)
    
    return recon_real, recon_zero_z, recon_rand_z


if __name__ == "__main__":
    print("Import this module and call:")
    print("  visualize_reconstruction(world_model, memory)")
    print("  visualize_bypass_test(world_model, memory)")
    print("  diagnose_encoder_embeddings(world_model, memory)")
    print("  diagnose_posterior_outputs(world_model, memory)")
    print("  diagnose_decoder_weights(world_model)")
    print("  test_z_sensitivity(world_model, memory)")
