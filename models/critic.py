import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class Critic(BaseModel):
    def __init__(self, feature_dim, hidden_dim, name='critic_network'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, 1)

        self.name = name

    def forward(self, x):
        # xu = torch.cat([state, action], 1)
        
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.output1(x)

        return x.squeeze(-1)
