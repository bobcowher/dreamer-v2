import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, feature_dim, hidden_dim, checkpoint_dir='checkpoints', name='critic_network'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)


    def forward(self, x):
        # xu = torch.cat([state, action], 1)
        
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        # x1 = F.relu(self.linear3(x1))
        x = self.output1(x)

        return x.squeeze(-1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, feature_dim, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='policy_network'):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)


    def forward(self, state):
        x = F.elu(self.linear1(state))
        x = F.elu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_dim = 1536
    action_dim = 3
    B = 4
    
    actor = Actor(feature_dim, action_dim, hidden_dim=512).to(device)
    critic = Critic(feature_dim, hidden_dim=512).to(device)
    
    features = torch.randn(B, feature_dim, device=device)
    
    # Test actor
    action, log_prob = actor.sample(features)
    print(f"action: {action.shape}")      # (4, 3)
    print(f"log_prob: {log_prob.shape}")  # (4, 1)
    
    # Test critic  
    value = critic(features)
    print(f"value: {value.shape}")        # (4,)
    
    print("\nActor-Critic OK!")
