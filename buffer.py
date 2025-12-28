import torch
import numpy as np
import os
import random
import sys

from torch._C import device

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions,
                 input_device, output_device='cpu', frame_stack=4):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device  = input_device
        
        print(f"Replay buffer memory on: {self.input_device}")

        self.output_device = output_device

        # States (uint8 saves ~4× RAM vs float32)
        self.state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )
        self.next_state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )

        # Actions as vectors for continuous action spaces
        self.action_memory  = torch.zeros((max_size, n_actions), dtype=torch.float32,
                                          device=self.input_device)
        self.reward_memory  = torch.zeros(max_size, dtype=torch.float32,
                                          device=self.input_device)
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool,
                                           device=self.input_device)

    # ------------------------------------------------------------------ #

    def can_sample(self, batch_size: int) -> bool:
        """Require at least 5×batch_size transitions before sampling."""
        return self.mem_ctr >= batch_size * 10

    # ------------------------------------------------------------------ #

    def store_transition(self, state, action, reward, next_state, done):
        """Write a transition in-place on `input_device`."""
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx]      = torch.as_tensor(
            state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(
            next_state, dtype=torch.uint8, device=self.input_device)

        self.action_memory[idx]   = torch.as_tensor(action, dtype=torch.float32, device=self.input_device)
        self.reward_memory[idx]   = float(reward)
        self.terminal_memory[idx] = bool(done)

        self.mem_ctr += 1

    # ------------------------------------------------------------------ #

    def sample_buffer(self, batch_size, sequence_length):
        """The goal here is to return a series of contiguous sequences.
           Returns:
               states: (B, T, C, H, W)
               actions: (B, T, N)
               rewards: (B, T)
               ...etc
        """

        max_mem = min(self.mem_ctr, self.mem_size)

        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        for _ in range(batch_size):
            while True:
                # Get 10 tries to get a batch without a done state. 
                start_pos = random.randint(0, max_mem - sequence_length - 1)
                batch   = torch.arange(start_pos, start_pos + sequence_length)

                states      = self.state_memory[batch]     \
                                .to(self.output_device, dtype=torch.uint8)
                actions     = self.action_memory[batch].to(self.output_device)
                rewards     = self.reward_memory[batch].to(self.output_device)
                next_states = self.next_state_memory[batch]\
                                .to(self.output_device, dtype=torch.uint8) 
                dones       = self.terminal_memory[batch].to(self.output_device)

                if dones.any():
                    pass
                    # Try again if there are any dones in the sequence.
                else:
                    batch_states.append(states)
                    batch_next_states.append(next_states)
                    batch_actions.append(actions)
                    batch_rewards.append(rewards)
                    batch_dones.append(dones)
                    break


        return (torch.stack(batch_states).to(self.output_device), 
                torch.stack(batch_actions).to(self.output_device), 
                torch.stack(batch_rewards).to(self.output_device), 
                torch.stack(batch_next_states).to(self.output_device), 
                torch.stack(batch_dones).to(self.output_device)
                )
    

    def print_stats(self):

        buffer_size = min(self.mem_ctr, self.mem_size)

        print(f"{buffer_size} memories in buffer.")
