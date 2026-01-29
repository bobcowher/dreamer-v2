import numpy as np
import gymnasium as gym
import cv2
# from gymnasium.spaces import sequence
import torch
import torch.nn.functional as F
from buffer import ReplayBuffer
from models.world_model import WorldModel
from models.actor import Actor
from models.critic import Critic
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import visualize
import random

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
        num_actions = self.env.action_space.shape[0] # pyright: ignore TODO: Fix this properly later 

        self.world_model = WorldModel(obs_shape=obs.shape, 
                                      action_dim=num_actions).to(self.device)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), learning_rate)

    #def __init__(self, feature_dim, num_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints', name='policy_network'):
        self.actor = Actor(feature_dim=feature_dim, 
                           num_actions=num_actions, 
                           hidden_dim=hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), learning_rate)

    # def __init__(self, feature_dim, hidden_dim, checkpoint_dir='checkpoints', name='critic_network'):
        self.critic = Critic(feature_dim=feature_dim, 
                             hidden_dim=hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), learning_rate)
        # print(self.env.action_space.shape)

        self.memory = ReplayBuffer(max_size=max_buffer_size, 
                                   input_shape=obs.shape, 
                                   n_actions=num_actions, 
                                   input_device=self.device, 
                                   output_device=self.device)
        
        self.left_bias = True

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        summary_writer_name = f'{summary_writer_name}_kl_weight_0.01_lg_network'
        self.summary_writer = SummaryWriter(summary_writer_name)
        self.total_steps_world_model = 0


    def __del__(self):
        self.env.close()

    def load_models(self):
        self.world_model.load_the_model(filename="world_model")
        self.critic.load_the_model(filename="critic")
        self.actor.load_the_model(filename="actor")

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

    def get_action(self, obs, h, z, deterministic=False):
        """
        Get action from policy given observation and RSSM state.
        
        Args:
            obs: processed observation (C, H, W)
            h: deterministic state (1, hidden_dim)
            z: stochastic state (1, stoch_flat)
            deterministic: if True, use mean instead of sampling
        
        Returns:
            action: numpy array scaled to env bounds
            h_new, z_new: updated RSSM state
        """
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
            embed = self.world_model.encode(obs_batch)
            
            # Dummy action for observe (action doesn't affect posterior much)
            dummy_action = torch.zeros(1, 1, 3, device=self.device)
            h, z, _, _ = self.world_model.observe(dummy_action, embed)
            h, z = h[:, -1], z[:, -1]
            
            features = torch.cat([h, z], dim=-1)
            
            if deterministic:
                mean, _ = self.actor.forward(features)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(features)
            
            action = action.squeeze(0).cpu().numpy()
        
        # Scale to environment bounds
        action[1] = (action[1] + 1) / 2  # gas: [-1,1] → [0,1]
        action[2] = (action[2] + 1) / 2  # brake: [-1,1] → [0,1]
        
        return action, h, z

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
            h, z = self.world_model.imagine_step(h, z, action)
            
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
      
        # Declaring avg loss. We'll add to this and average it at the end. 
        avg_loss = {"world_model": 0.0,
                      "recon": 0.0,
                      "reward": 0.0,
                      "continue": 0.0,
                      "kl": 0.0}

        for _ in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size, sequence_length)
           
            continues = 1.0 - dones.float()  # Convert dones to continues
            
            loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, continues)
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()

            avg_loss["world_model"] += loss.item()
            avg_loss["recon"]       += loss_dict["recon"]
            avg_loss["reward"]      += loss_dict["reward"]
            avg_loss["continue"]    += loss_dict["continue"]
            avg_loss["kl"]          += loss_dict["kl"]
            
        # Actually average average loss
        for key, val in avg_loss.items():
            avg_loss[key] = val / epochs

        return avg_loss

    
    def evaluate_policy(self, num_episodes=3):
        total_reward = 0
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)
            done = False
            episode_reward = 0
            
            h, z = self.world_model.get_initial_state(1)
            
            while not done:
                action, h, z = self.get_action(obs, h, z)

                
                obs, reward, done, truncated, _ = self.env.step(action)
                obs = self.process_observation(obs)
                done = done or truncated
                episode_reward += float(reward)
                
                if done:
                    # Only print the last action, for log purposes. 
                    print(f"Action: steer={action[0]:.3f}, gas={action[1]:.3f}, brake={action[2]:.3f}")
            
            total_reward += episode_reward
        
        return total_reward / num_episodes

                
    def train_actor_critic(self, epochs=1, batch_size=16, horizon=15):
        """
        Train actor and critic on imagined trajectories.
        
        Returns:
            dict with actor_loss, critic_loss
        """

        total_critic_loss = 0
        total_actor_loss = 0

        for epoch in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size, 1)

            embeds = self.world_model.encode(obs)

            h, z, _, _ = self.world_model.observe(actions, embeds)

            h_final = h[:, -1]
            z_final = z[:, -1]

            features, actions, log_probs = self.imagine_trajectory(h_final, z_final, horizon)

            batch_size, horizon_plus_1, feature_dim = features.shape
            horizon = horizon_plus_1 - 1

            # Flatten for MLPs, then restore shape
            # Don't let the actor gradients flow backwards through the reward predictor
            with torch.no_grad():
                rewards = self.world_model.reward_pred(
                    features[:, 1:].reshape(batch_size * horizon, feature_dim)
                ).reshape(batch_size, horizon)

            values = self.critic(
                features.reshape(batch_size * horizon_plus_1, feature_dim)
            ).reshape(batch_size, horizon_plus_1)

            with torch.no_grad():
                returns = self.compute_lambda_returns(rewards, values.detach())
                advantage = returns - values[:, :-1].detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            critic_loss = F.mse_loss(values[:, :-1], returns.detach())
            actor_loss = -(log_probs * advantage).mean()

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            (actor_loss + critic_loss).backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100)

            self.critic_optimizer.step()
            self.actor_optimizer.step()

        avg_critic_loss = total_critic_loss / epochs
        avg_actor_loss = total_actor_loss / epochs

        return avg_actor_loss, avg_critic_loss


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

            loss = F.l1_loss(obs_flat, recon)
            
            total_loss += loss.item()
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()
            
        avg_loss = total_loss / epochs

        return avg_loss



    def collect_dataset(self, 
                        episodes : int,
                        use_policy=True):

        total_reward = 0

        for episode in range(episodes):
            done = False
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            episode_reward = 0.0

            h, z = self.world_model.get_initial_state(1)
            
            while not done:
                if use_policy:
                    action, h, z = self.get_action(obs, h, z)
                else:
                    action = self.heuristic_action()
                
                next_obs, reward, done, truncated, info  = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                
                done = done or truncated    

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                if(random.random() < 0.01):
                    print(f"Action Sample: {action}")

                episode_reward = episode_reward + float(reward)

            total_reward += episode_reward
            
            self.left_bias = not self.left_bias

        return total_reward / episodes

            # if use_policy == False:
            #     print(f"Heuristic policy episode reward: {episode_reward}")
            #

    def train(self, epochs=0):

        self.collect_dataset(50, use_policy=False)

        for epoch in range(epochs):
            live_reward = self.collect_dataset(1)
            world_model_loss = self.train_world_model(epochs=10, batch_size=128, sequence_length=50)
            #
            # loss = self.train_encoder(epochs=50, batch_size=16, sequence_length=16)
            visualize.visualize_reconstruction(self.world_model, self.memory, num_samples=4)

            actor_loss, critic_loss = self.train_actor_critic(epochs=5)

            # if(epoch % 10 == 0):
            #     reward = self.evaluate_policy()
            print(f"Epoch {epoch} Eval Reward: {live_reward}")
            self.summary_writer.add_scalar("Eval/Reward", live_reward, epoch)

            print(f"Epoch {epoch} Loss - World Model: {world_model_loss} Actor: {actor_loss} Critic: {critic_loss}")

            self.summary_writer.add_scalar("Loss/Actor", actor_loss, epoch)
            self.summary_writer.add_scalar("Loss/Critic", critic_loss, epoch)

            for key, val in world_model_loss.items():
                self.summary_writer.add_scalar(f"Loss/WorldModel {key}", val, epoch)

            self.world_model.save_the_model(filename="world_model")
            self.critic.save_the_model(filename="critic")
            self.actor.save_the_model(filename="actor")
    

        
    

