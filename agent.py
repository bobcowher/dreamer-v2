import gymnasium as gym
import cv2
import torch
import numpy as np
from buffer import ReplayBuffer
from model import Encoder
import torch.nn.functional as F

class Agent:

    def __init__(self, render_mode="human",
                       max_buffer_size=10000,
                       learning_rate=0.0001):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=True)

        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        observation, info = self.env.reset(seed=42)
        
        self.encoder = Encoder(observation_shape=obs.shape).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), learning_rate) 

        # print(self.env.action_space.shape)

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.shape[0], input_device=self.device, output_device=self.device)
        
        self.left_bias = True

        pass

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
    

    def train_encoder(self,
                  epochs : int,
                  batch_size: int,
                  sequence_length: int):

        total_recon_loss = 0
        total_loss       = 0

        for epoch in range(epochs):

            # 1 — sample & reshape
            obs, _, _, _, _ = self.memory.sample_buffer(batch_size, sequence_length)

            obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:]) 
            # 2 — Q(s,a) with the online network
            recon, embed = self.encoder(obs_flat)

            # print("Obs:", type(observations))
            # print("Pred:", type(predicted_observations))
            #
            # 4 — loss & optimise
            # Normalize observations for comparison with reconstruction
            obs_normalized = obs_flat.float() / 255.0
            # print(f"Recon: {type(recon)}")
            # print(f"Obs Norm: {type(obs_normalized)}")
            loss = F.mse_loss(obs_normalized, recon)
            # writer.add_scalar("Stats/model_loss", loss.item(), total_steps)

            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()

            print(f"Encoder Loss {loss.item()}")

            total_loss += loss.item()

        ave_loss = total_loss / epochs

        return ave_loss


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

            print(f"Completed episode {episode} with score {episode_reward}")

            self.memory.print_stats()


    def train(self, epochs=0):

        for _ in range(epochs):
            self.collect_dataset(50)
            loss, recon_loss = self.train_encoder(50, batch_size=16, sequence_length=16)

            print(f"Loss: {loss}, Recon Loss: {recon_loss}")



            
        

    def get_action(self):
        pass


