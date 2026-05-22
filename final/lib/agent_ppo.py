import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):

    def __init__(self, num_inputs: int, num_actions: int, hidden_size: int = 512):
        super(PPOAgent, self).__init__()

        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh()
        )
        
        
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)
        

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
    
    # weight initialization
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        
        if action is None:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean(-1)
        
        return action, log_prob, entropy, self.get_value(x)
