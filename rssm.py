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

    def initial_state(self, batch_size: int, device: torch.device):
        """Return zeros for (h, z)."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_flat, device=device)
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
        x = self.pre_gru(torch.cat([h, z, action]))

        h_new = self.gru(x, h)
        prior_logits = self.prior_net(h_new)

        post_logits = self.post_net(torch.cat([h_new, embed]), dim=-1)

        z_new = self.sample_stochastic(post_logits) 
        
        return h_new, z_new, prior_logits, post_logits



    def imagine_step(self, h, z, action):
        """Single step without observation (imagination)."""
        pass
