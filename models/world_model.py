import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.rssm import RSSM, RewardPredictor, ContinuePredictor
from models.base import BaseModel


class WorldModel(BaseModel):
    def __init__(
        self,
        obs_shape,              # (3, 64, 64)
        action_dim,             # 3 for CarRacing
        gru_hidden_dim=128,     # GRU state size (SMALL - bottleneck)
        mlp_hidden_dim=512,     # MLP hidden layers (LARGE - capacity)
        stoch_dim=32,
        stoch_classes=32,
        embed_dim=1024,         # Match encoder output
    ):
        super().__init__()
        
        self.gru_hidden_dim = gru_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        
        self.encoder = Encoder(observation_shape=obs_shape, embed_dim=embed_dim)
        
        # Feature dimension for decoder: h + z
        self.stoch_flat = stoch_dim * stoch_classes
        feature_dim = gru_hidden_dim + self.stoch_flat  # 128 + 1024 = 1152
        
        self.decoder = Decoder(observation_shape=obs_shape, embed_dim=feature_dim)

        self.embed_dim = embed_dim

        self.rssm = RSSM(
            action_dim=action_dim, 
            embed_dim=embed_dim, 
            gru_hidden_dim=gru_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            stoch_dim=stoch_dim, 
            stoch_classes=stoch_classes
        )
        
        self.reward_pred = RewardPredictor(feature_dim, hidden_dim=mlp_hidden_dim)
        self.continue_pred = ContinuePredictor(feature_dim, hidden_dim=mlp_hidden_dim)

    def save_the_model(self, filename="world_model", verbose=False):
        self.encoder.save_the_model(filename="encoder")
        self.decoder.save_the_model(filename="decoder")
        self.rssm.save_the_model(filename="rssm")
        return super().save_the_model(filename=filename, verbose=verbose)

    def load_the_model(self, filename="world_model", device='cuda'):
        self.encoder.load_the_model(filename="encoder")
        self.decoder.load_the_model(filename="decoder")
        self.rssm.load_the_model(filename="rssm")
        return super().load_the_model(filename, device)
    
    def forward(self, obs, actions):
        """
        Args:
            obs:     (B, T, C, H, W)
            actions: (B, T, action_dim)
        Returns:
            Dict with states, predictions, logits
        """
        embeds = self.encode(obs)

        # Run RSSM
        h_all, z_all, prior_all, post_all = self.rssm.observe_sequence(actions, embeds)
        
        # Get features for predictors
        features = torch.cat([h_all, z_all], dim=-1)  # (B, T, feature_dim)
        
        # Predictions
        reward_pred = self.reward_pred(features)
        continue_pred = self.continue_pred(features)
        
        return {
            "h": h_all,
            "z": z_all,
            "prior_logits": prior_all,
            "post_logits": post_all,
            "reward_pred": reward_pred,
            "continue_pred": continue_pred,
            "embeds": embeds,
        }


    def encode(self, obs):
        batch_size, sequence_length = obs.shape[:2]
        obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:])
        embed_flat = self.encoder(obs_flat)
        embeds = embed_flat.view(batch_size, sequence_length, -1)

        return embeds


    def kl_divergence(self, prior_logits, post_logits, alpha=0.8, free_nats=1.0):
        # Use model dims, not hard-coded 32s
        stoch_dim = self.rssm.stoch_dim
        stoch_classes = self.rssm.stoch_classes

        prior_logits = prior_logits.view(-1, stoch_dim, stoch_classes)
        post_logits  = post_logits.view(-1, stoch_dim, stoch_classes)

        prior_logprob = F.log_softmax(prior_logits, dim=-1)
        post_logprob  = F.log_softmax(post_logits,  dim=-1)

        prior_prob = prior_logprob.exp()
        post_prob  = post_logprob.exp()

        # KL per variable: (B*T, stoch_dim)
        kl_per_var = (post_prob * (post_logprob - prior_logprob)).sum(dim=-1)

        # KL balancing
        kl_prior = (post_prob.detach() * (post_logprob.detach() - prior_logprob)).sum(dim=-1)
        kl_post  = (post_prob * (post_logprob - prior_logprob.detach())).sum(dim=-1)

        # Free nats per variable
        kl_prior = torch.clamp(kl_prior - free_nats, min=0.0)
        kl_post  = torch.clamp(kl_post  - free_nats, min=0.0)

        kl = alpha * kl_prior + (1 - alpha) * kl_post

        # Sum vars, mean batch/time
        return kl.sum(dim=-1).mean()
    

    def compute_loss(self, obs, actions, rewards, continues, kl_weight=0.01):
        """
        Args:
            obs:       (B, T, C, H, W) uint8
            actions:   (B, T, action_dim)
            rewards:   (B, T)
            continues: (B, T) — 1.0 if episode continues, 0.0 if done
        
        Returns:
            total_loss, loss_dict
        """
        outputs = self.forward(obs, actions)
        
        batch_size, seq_len = obs.shape[:2]
        
        # 1. Reconstruction loss — decode from (h, z), compare to obs
        features = torch.cat([outputs["h"], outputs["z"]], dim=-1)
        features_flat = features.view(batch_size * seq_len, -1)
        recon = self.decoder(features_flat)
        recon = recon.view(batch_size, seq_len, *obs.shape[2:])
        
        obs_normalized = obs.float() / 255.0
        recon_loss = F.l1_loss(recon, obs_normalized)
        
        # 2. Reward loss
        reward_loss = F.mse_loss(outputs["reward_pred"], rewards)
        
        # 3. Continue loss
        continue_loss = F.binary_cross_entropy_with_logits(
            outputs["continue_pred"], continues
        )
        
        # 4. KL loss (prior vs posterior)
        kl_loss = self.kl_divergence(
            outputs["prior_logits"], 
            outputs["post_logits"]
        )

        total_loss = recon_loss + reward_loss + continue_loss + (kl_weight * kl_loss)
        
        return total_loss, {
            "recon": recon_loss.item(),
            "reward": reward_loss.item(),
            "continue": continue_loss.item(),
            "kl": kl_loss.item(),
            "total": total_loss.item(),
        }

    def get_initial_state(self, batch_size):
        return self.rssm.initial_state(batch_size)

    def observe(self, actions, embeds):
        return self.rssm.observe_sequence(actions, embeds)

    def imagine_step(self, h, z, action):
        return self.rssm.imagine_step(h, z, action)

    def predict_reward(self, features):
        return self.reward_pred(features)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config
    obs_shape = (3, 64, 64)
    action_dim = 3
    B, T = 4, 16
    
    # Create model with new architecture
    model = WorldModel(
        obs_shape, 
        action_dim,
        gru_hidden_dim=128,   # Small bottleneck
        mlp_hidden_dim=512,   # Large capacity
    ).to(device)
    
    print(f"WorldModel created on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GRU hidden (bottleneck): {model.gru_hidden_dim}")
    print(f"MLP hidden (capacity):   {model.mlp_hidden_dim}")
    print(f"Feature dim (h+z):       {model.gru_hidden_dim + model.stoch_flat}")
    
    # Fake batch
    obs = torch.randint(0, 256, (B, T, *obs_shape), dtype=torch.uint8, device=device)
    actions = torch.randn(B, T, action_dim, device=device)
    rewards = torch.randn(B, T, device=device)
    continues = torch.ones(B, T, device=device)
    
    # Test forward
    print("\n--- Testing forward ---")
    outputs = model(obs, actions)
    print(f"h: {outputs['h'].shape}")              # (4, 16, 128)
    print(f"z: {outputs['z'].shape}")              # (4, 16, 1024)
    print(f"reward_pred: {outputs['reward_pred'].shape}")  # (4, 16)
    
    # Test loss
    print("\n--- Testing compute_loss ---")
    loss, loss_dict = model.compute_loss(obs, actions, rewards, continues)
    print(f"Losses: {loss_dict}")
    
    # Test gradients
    loss.backward()
    print(f"Gradients OK: {model.encoder.fc_enc.weight.grad is not None}")
