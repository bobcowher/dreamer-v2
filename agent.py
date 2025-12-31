import gymnasium as gym
import cv2
from gymnasium.spaces import sequence
import torch
import numpy as np
from buffer import ReplayBuffer
from encoder import Encoder
from world_model import WorldModel
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from actor_critic import Actor, Critic
import datetime
import visualize

class Agent:

    def __init__(self, render_mode="human",
                       max_buffer_size=10000,
                       learning_rate=0.0001):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=True)

        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        observation, info = self.env.reset(seed=42)

        feature_dim = 1536
        hidden_dim = 512
        
        # self.encoder = Encoder(observation_shape=obs.shape).to(self.device)
        # self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), learning_rate) 
        num_actions = self.env.action_space.shape[0] 

        self.world_model = WorldModel(obs_shape=obs.shape, 
                                      action_dim=num_actions).to(self.device)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), learning_rate)

    #def __init__(self, feature_dim, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='policy_network'):
        self.actor = Actor(feature_dim=feature_dim, 
                           num_actions=num_actions, 
                           hidden_dim=hidden_dim).to(self.device)

    # def __init__(self, feature_dim, hidden_dim, checkpoint_dir='checkpoints', name='critic_network'):
        self.critic = Critic(feature_dim=feature_dim, 
                             hidden_dim=hidden_dim).to(self.device)
        # print(self.env.action_space.shape)

        self.memory = ReplayBuffer(max_size=max_buffer_size, 
                                   input_shape=obs.shape, 
                                   n_actions=num_actions, 
                                   input_device=self.device, 
                                   output_device=self.device)
        
        self.left_bias = True

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.summary_writer = SummaryWriter(summary_writer_name)
        self.total_steps = 0


    def __del__(self):
        self.env.close()

    def heuristic_action(self):
        
        if(self.left_bias):
            left_steer = 0.03
            right_steer = 0.01
        else:
            left_steer = 0.01
            right_steer = 0.03

        steer = np.random.normal(left_steer, right_steer)     # small right bias
        gas   = 0.5
        brake = 0.0
        return np.array([steer, gas, brake], dtype=np.float32)
    
    def process_observation(self, obs):
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1).to(self.device)
        return obs 

    def imagine_trajectory(self, start_h, start_z, horizon):
        """
        Args:
            start_h: (B, hidden_dim) — initial deterministic state
            start_z: (B, stoch_flat) — initial stochastic state
            horizon: int — steps to imagine
        
        Returns:
            features: (B, H+1, feature_dim) — all states including start
            actions:  (B, H, action_dim)
            log_probs: (B, H)
        """
        h, z = start_h, start_z
        
        h_list = [h]
        z_list = [z]
        action_list = []
        log_prob_list = []
        
        for _ in range(horizon):
            features = torch.cat([h, z], dim=-1)
            action, log_prob = self.actor.sample(features)
            h, z = self.world_model.rssm.imagine_step(h, z, action)
            
            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            log_prob_list.append(log_prob)

       
        h_all = torch.stack(h_list, dim=1)
        z_all = torch.stack(z_list, dim=1)
        features = torch.cat([h_all, z_all], dim=-1)
        actions = torch.stack(action_list, dim=1)
        log_probs = torch.stack(log_prob_list, dim=1)

        return features, actions, log_probs
        
    # Stack into tensors
    # features: concat all (h, z) pairs
    # Return shapes for downstream loss computation
    def compute_lambda_returns(self, rewards, values, gamma=0.99, lambda_=0.95):
        """
        Args:
            rewards: (B, H) — predicted rewards from imagination
            values:  (B, H+1) — predicted values (includes final state)
        
        Returns:
            returns: (B, H) — λ-return targets for each timestep
        """
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        
        # Bootstrap from final value
        next_return = values[:, -1]
        
        # Work backwards
        for t in reversed(range(H)):
            next_return = rewards[:, t] + gamma * (
                (1 - lambda_) * values[:, t+1] + lambda_ * next_return
            )
            returns[:, t] = next_return
    
        return returns
    

    def train_world_model(self, epochs, batch_size, sequence_length):
       
        total_loss = 0

        for _ in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size, sequence_length)
           
            continues = 1.0 - dones.float()  # Convert dones to continues
            
            loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, continues)
            
            total_loss += loss.item()
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()
            
            self.summary_writer.add_scalar("Loss/recon", loss_dict["recon"], self.total_steps)
            self.summary_writer.add_scalar("Loss/reward", loss_dict["reward"], self.total_steps)
            self.summary_writer.add_scalar("Loss/continue", loss_dict["continue"], self.total_steps)
            self.summary_writer.add_scalar("Loss/kl", loss_dict["kl"], self.total_steps)

            self.total_steps += 1

        avg_loss = total_loss / epochs

        return avg_loss
            


    def train_encoder(self,
                  epochs : int,
                  batch_size: int,
                  sequence_length: int):
        total_loss = 0

        for _ in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size, sequence_length)
           
            obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:]).float()

            continues = 1.0 - dones.float()  # Convert dones to continues

            embed = self.world_model.encoder(obs_flat)
            padded = F.pad(embed, (0, 512))  # 1024 → 1536
            recon = self.world_model.decoder(padded)

            loss = F.mse_loss(obs_flat, recon)
            
            total_loss += loss.item()
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()
            
            self.total_steps += 1

        avg_loss = total_loss / epochs

        return avg_loss



    def collect_dataset(self, 
                        episodes : int):

        for episode in range(episodes):
            done = False
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            episode_reward = 0.0

            
            while not done:
                action = self.heuristic_action()
                
                next_obs, reward, done, truncated, info  = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                
                done = done or truncated    

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                episode_reward = episode_reward + float(reward)
            
            self.left_bias = not self.left_bias

            # print(f"Completed episode {episode} with score {episode_reward}")

            # self.memory.print_stats()


    def train(self, epochs=0):

        for _ in range(epochs):
            self.collect_dataset(10)
            loss = self.train_world_model(epochs=50, batch_size=16, sequence_length=16)
            #
            # loss = self.train_encoder(epochs=50, batch_size=16, sequence_length=16)
            print(f"Loss: {loss}")
            visualize.visualize_reconstruction(self.world_model, self.memory, num_samples=4)


            
        

    def get_action(self):
        pass


