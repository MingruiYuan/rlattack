import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def init_weights(layer):
    if hasattr(layer, 'weight'):
        if len(layer.weight.shape) > 1:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class highwayConv(nn.Module):
    """
    This is a highway convolution layer which is used in some models.
    It will use "same" convolution.
    """

    def __init__(self, dimension, kernel_size, dilation, causal=False):
        """
        Args:
        --dimension. Number of input and output channels. They always keep same in highway nets.
        --causal. A boolean on whether to use causal convolution or ordinary one.
        """
        super(highwayConv, self).__init__()
        self.dimension = dimension
        self.causal = causal

        # Same Convolution. Stride = 1.
        # L_out = L_in + 2*padding - dilation*(kernel_size - 1)
        # As in this model, kernel_size is always odd,
        # we do not consider the case that pad is not an integer.
        self.pad = dilation*(kernel_size-1) // 2

        self.conv = nn.Conv1d(in_channels=dimension, out_channels=2*dimension, kernel_size=kernel_size, padding=0 if causal else self.pad, dilation=dilation)
        self.ln1 = nn.LayerNorm(normalized_shape=dimension)
        self.ln2 = nn.LayerNorm(normalized_shape=dimension)

    def forward(self, inputs):
        """
        --inputs: (Batch, dimension, timeseries)
        --x: (Batch, 2*dimension, timeseries)
        --H1/H2: (Batch, dimension, timeseries)
        --outputs: (Batch, dimension, timeseries)
        """

        # zero-padding prior to the inputs to ensure causal convolution.
        if self.causal and self.pad>0:
            d1, d2, d3 = inputs.size()
            inputs = torch.cat((torch.zeros((d1, d2, 2*self.pad)).to(device), inputs), dim=-1)

        #print('hc inputs device: ', inputs.device)

        x = self.conv(inputs)
        H1 = x[:, :self.dimension, :]
        H2 = x[:, self.dimension:, :]
        H1 = self.ln1(H1.permute(0,2,1)).permute(0,2,1)
        H2 = self.ln2(H2.permute(0,2,1)).permute(0,2,1)
        outputs = F.sigmoid(H1)*H2+(1-F.sigmoid(H1))*inputs[:, :, 2*self.pad if self.causal else 0:]
        return outputs

class highwayDilationIncrement(nn.Module):

    def __init__(self, dimension, causal=False):
        """
        --causal. A boolean on whether to use causal convolution or ordinary one.
        """
        super(highwayDilationIncrement, self).__init__()

        self.hc1 = highwayConv(dimension=dimension, kernel_size=3, dilation=1, causal=causal)
        self.hc2 = highwayConv(dimension=dimension, kernel_size=3, dilation=3, causal=causal)
        self.hc3 = highwayConv(dimension=dimension, kernel_size=3, dilation=9, causal=causal)

    def forward(self, inputs):
        x = self.hc1(inputs)
        x = self.hc2(x)
        x = self.hc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, norm_scale):
        super(Actor, self).__init__()
        # Need an adversarial factor to constrain the L_inf norm of perturbation.
        self.norm_scale = norm_scale
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.hci1 = highwayDilationIncrement(dimension=hidden_dim)
        self.hci2 = highwayDilationIncrement(dimension=hidden_dim)
        self.hc = highwayConv(dimension=hidden_dim, kernel_size=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)
        self.ln2 = nn.LayerNorm(normalized_shape=input_dim)

    def forward(self, state):
        x = self.conv1(state)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.hci1(F.relu(x))
        x = self.hci2(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.norm_scale*F.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, linlayer_dim):
        super(Critic, self).__init__()
        self.conv_state = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.ln1_state = nn.LayerNorm(normalized_shape=hidden_dim)
        self.conv_action = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.ln1_action = nn.LayerNorm(normalized_shape=hidden_dim)
        self.hci = highwayDilationIncrement(dimension=2*hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=2*hidden_dim, out_channels=8, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=4)
        self.ln2 = nn.LayerNorm(normalized_shape=8)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
        self.linear = nn.Linear(in_features=linlayer_dim, out_features=1)

    def forward(self, state, action):
        x = self.conv_state(state)
        x = self.ln1_state(x.permute(0,2,1)).permute(0,2,1)
        y = self.conv_action(action)
        y = self.ln1_action(y.permute(0,2,1)).permute(0,2,1)
        x = self.hci(F.relu(torch.cat((x, y), dim=1)))
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv3(F.relu(x))
        x = self.linear(x).squeeze(dim=-1)
        return x

class ActionNoise(object):
    def __init__(self, *size, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.ones(size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.size)
        self.state = x + dx
        return self.state * self.scale

class DDPG(object):
    def __init__(self, cfg, ft):

        self.norm_scale = 1.0 if ft == 'LFCC' else 0.4
        self.ft = ft
        self.actor = Actor(cfg['MEL_DIM'], cfg['AC_HIDDEN_DIM'], self.norm_scale)
        self.actor.to(device)
        self.actor_tar = Actor(cfg['MEL_DIM'], cfg['AC_HIDDEN_DIM'], self.norm_scale)
        self.actor_tar.to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg['AC_LR'])

        self.critic = Critic(cfg['MEL_DIM'], cfg['CR_HIDDEN_DIM'], cfg['LINLAYER_DIM'])
        self.critic.to(device)
        self.critic_tar = Critic(cfg['MEL_DIM'], cfg['CR_HIDDEN_DIM'], cfg['LINLAYER_DIM'])
        self.critic_tar.to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg['CR_LR'])

        self.cfg = cfg

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(state)

        self.actor.train()
        mu = mu.detach().squeeze()

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(device)

        return mu.clamp(-self.norm_scale, self.norm_scale)

    def update_parameters(self, batch):
        state = torch.stack(batch.state, dim=0).to(device)
        action = torch.stack(batch.action, dim=0).to(device)
        reward = torch.stack(batch.reward, dim=0).to(device)
        mask = torch.stack(batch.mask, dim=0).to(device)
        next_state = torch.stack(batch.next_state, dim=0).to(device)

        next_action = self.actor_tar(next_state)
        next_state_action_values = self.critic_tar(next_state, next_action)
        expected_values = reward + (self.cfg['GAMMA']*mask*next_state_action_values)

        self.critic_opt.zero_grad()
        network_values = self.critic(state, action)
        value_loss = F.mse_loss(network_values, expected_values)
        value_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        policy_loss = -self.critic(state, self.actor(state)).mean()
        policy_loss.backward()
        self.actor_opt.step()

        soft_update(self.actor_tar, self.actor, self.cfg['TAU'])
        soft_update(self.critic_tar, self.critic, self.cfg['TAU'])

        return value_loss.item(), policy_loss.item()

    def save_model(self, ctime, update):
        save_dir = self.cfg['ROOT_DIR']+'saved_models/attack/{}/'.format(ctime)
        if not os.path.exists(save_dir):
            os.system('mkdir -p '+save_dir)

        torch.save(self.actor.state_dict(), save_dir+'actor_upd{}.pt'.format(str(update+1)))
        torch.save(self.critic.state_dict(), save_dir+'critic_upd{}.pt'.format(str(update+1)))

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        else:
            self.actor.load_state_dict(torch.load(self.cfg['ROOT_DIR']+'saved_models/initial_models/initial_actor_{}.pt'.format(self.ft), map_location=device))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path, map_location=device))
        else:
            self.actor.load_state_dict(torch.load(self.cfg['ROOT_DIR']+'saved_models/initial_models/initial_critic_{}.pt'.format(self.ft), map_location=device))
            
        hard_update(self.actor_tar, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_tar, self.critic)