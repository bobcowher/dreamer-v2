# Dreamer V2 Variable Legend

## Tensor Dimensions

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| B | Batch size | 32 |
| T | Timesteps / sequence length | 16 |
| C | Channels (RGB) | 3 |
| H | Height | 64 |
| W | Width | 64 |

## State Variables

| Symbol | Meaning | Shape |
|--------|---------|-------|
| h | Deterministic state (GRU hidden) | (B, 512) |
| z | Stochastic state (categorical, flattened) | (B, 1024) |
| a | Action | (B, action_dim) |
| e / embed | Encoded observation | (B, 1024) |
| obs | Raw observation | (B, C, H, W) |
| recon | Reconstructed observation | (B, C, H, W) |

## Dimension Sizes (Dreamer V2 Defaults)

| Name | Value | Notes |
|------|-------|-------|
| hidden_dim | 512 | GRU state size |
| stoch_dim | 32 | Number of categorical variables |
| stoch_classes | 32 | Classes per variable |
| stoch_flat | 1024 | 32 × 32, flattened z |
| embed_dim | 1024 | Encoder output |
| feature_dim | 1152 | h + z concatenated |

## Shape Examples

```
Single timestep:
    obs:    (B, C, H, W)     = (32, 3, 64, 64)
    embed:  (B, embed_dim)   = (32, 1024)
    h:      (B, hidden_dim)  = (32, 512)
    z:      (B, stoch_flat)  = (32, 1024)
    
Sequence:
    obs:    (B, T, C, H, W)  = (32, 16, 3, 64, 64)
    embed:  (B, T, embed_dim)= (32, 16, 1024)
    h:      (B, T, hidden_dim)= (32, 16, 512)
    z:      (B, T, stoch_flat)= (32, 16, 1024)
```
