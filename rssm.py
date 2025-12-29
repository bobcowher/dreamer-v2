import torch
import torch.nn as nn
import torch.nn.functional as F

# h = previous deterministic state
# z = new stochastic state (sampled from posterior)
# a = action
# e = encoded observation. 

class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int = 1024,      # From encoder
        hidden_dim: int = 512,       # GRU hidden size
        stoch_dim: int = 32,         # Categorical variables
        stoch_classes: int = 32,     # Classes per variable
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_dim * stoch_classes  # 1024

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # TODO: Define layers
        # 1. pre_gru: projects (h, z, a) → GRU input
        self.pre_gru = nn.Sequential(
            nn.Linear(hidden_dim + self.stoch_flat + action_dim, hidden_dim),
            nn.ELU(),
        )
        # 2. gru: GRUCell
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # 3. prior_net: h → logits
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )
        # 4. posterior_net: (h, embed) → logits
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

    def initial_state(self, batch_size: int):
        """Return zeros for (h, z)."""
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        z = torch.zeros(batch_size, self.stoch_flat, device=self.device)
        return h, z


    def sample_stochastic(self, logits):
        """
        logits: (B, stoch_flat) → (B, 32*32)
        returns: (B, stoch_flat) one-hot samples
        """
        B = logits.shape[0]
        
        # Reshape to (B, 32 variables, 32 classes)
        logits = logits.view(B, self.stoch_dim, self.stoch_classes)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample one-hot (non-differentiable)
        dist = torch.distributions.OneHotCategorical(probs=probs)
        samples = dist.sample()  # (B, 32, 32) one-hot
        
        # Straight-through: samples in forward, but gradient flows through probs
        samples = samples + probs - probs.detach()
        
        # Flatten back
        return samples.view(B, -1)  # (B, 1024)


    def observe_step(self, h, z, action, embed):
        """Single step with observation (training)."""
        x = self.pre_gru(torch.cat([h, z, action], dim=-1))

        h_new = self.gru(x, h)
        prior_logits = self.prior_net(h_new)

        post_logits = self.post_net(torch.cat([h_new, embed], dim=-1))

        z_new = self.sample_stochastic(post_logits) 
        
        return h_new, z_new, prior_logits, post_logits


    def observe_sequence(self, actions, embeds):
        """
        Process a full sequence with observations.
        
        Args:
            actions: (B, T, action_dim)
            embeds:  (B, T, embed_dim)
        
        Returns:
            h_all:         (B, T, hidden_dim)
            z_all:         (B, T, stoch_flat)
            prior_all:     (B, T, stoch_flat)
            posterior_all: (B, T, stoch_flat)
        """
        batch_size = 32

        h, z = self.initial_state(batch_size=batch_size)


    def imagine_step(self, h, z, action):
        """Single step without observation (imagination)."""
        x = self.pre_gru(torch.cat([h, z, action], dim=-1))

        h_new = self.gru(x, h)
        prior_logits = self.prior_net(h_new)

        z_new = self.sample_stochastic(prior_logits) 
        
        return h_new, z_new 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config
    B = 4           # batch size
    action_dim = 3  # CarRacing: steer, gas, brake
    embed_dim = 1024
    
    rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim).to(device)
    
    # Init state
    h, z = rssm.initial_state(B, device)
    print(f"h: {h.shape}, z: {z.shape}")
    # Expected: h: (4, 512), z: (4, 1024)
    
    # Fake inputs
    action = torch.randn(B, action_dim, device=device)
    embed = torch.randn(B, embed_dim, device=device)
    
    # Test observe_step
    h_new, z_new, prior, post = rssm.observe_step(h, z, action, embed)
    print(f"h_new: {h_new.shape}, z_new: {z_new.shape}")
    print(f"prior: {prior.shape}, post: {post.shape}")
    # Expected: all (4, 512) or (4, 1024)
    
    # Test imagine_step
    h_imag, z_imag = rssm.imagine_step(h, z, action)
    print(f"h_imag: {h_imag.shape}, z_imag: {z_imag.shape}")
    
    # Test gradients flow
    loss = z_new.sum()
    loss.backward()

    print(f"Gradients OK: {rssm.post_net[0].weight.grad is not None}")
    
    print("\nAll shapes correct!")
    # Test observe_sequence
    print("\n--- Testing observe_sequence ---")
    B, T = 4, 16
    actions = torch.randn(B, T, action_dim, device=device)
    embeds = torch.randn(B, T, embed_dim, device=device)

    h_all, z_all, prior_all, post_all = rssm.observe_sequence(actions, embeds)

    print(f"h_all: {h_all.shape}")       # Expected: (4, 16, 512)
    print(f"z_all: {z_all.shape}")       # Expected: (4, 16, 1024)
    print(f"prior_all: {prior_all.shape}")   # Expected: (4, 16, 1024)
    print(f"post_all: {post_all.shape}")     # Expected: (4, 16, 1024)

    # Verify gradients flow through time
    loss = z_all.sum()
    loss.backward()
    print(f"Gradients through sequence: {rssm.gru.weight_hh.grad is not None}")
