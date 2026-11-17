#!/usr/bin/env python3
"""
Autonomous Self-Parking Car Simulation with PyTorch Deep Q-Network (DQN)
Integrates deep reinforcement learning with the ASCII terminal visualizer.

To run with UV:
    uv add torch numpy
    uv run python autopark_pytorch.py
"""

import sys
import os
import time
import random
import argparse
from collections import deque

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("\n\033[91m[!] PyTorch or NumPy is not installed.\033[0m")
    print("    Install them with UV:")
    print("      \033[92muv add torch numpy\033[0m")
    print("    Or run directly:")
    print("      \033[92muv run python autopark_pytorch.py\033[0m\n")
    sys.exit(1)

from autopark import (
    Color,
    ParkingEnvironment,
    AsciiVisualizer,
    DIR_NAMES,
    run_animated_episode,
    multi_spot_challenge,
    print_banner,
)


# -----------------------------------------------------------------------------
# PyTorch Deep Q-Network Architecture
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Multi-layer Perceptron (MLP) approximating the Q-value function:
    
    Mathematical Formulation:
        Q_theta(s, a) ≈ Q*(s, a)
        
    Takes a 12-dimensional continuous feature representation of the car's state
    and outputs estimated Q-values for each discrete action:
        0: FORWARD, 1: REVERSE, 2: TURN_LEFT, 3: TURN_RIGHT
    """
    def __init__(self, state_dim=12, action_dim=4):
        """
        Initializes the feed-forward neural network layers.
        
        Args:
            state_dim (int): Dimensionality of the input feature vector (default: 12).
            action_dim (int): Number of discrete output actions (default: 4).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x (torch.Tensor): State feature tensor of shape (batch_size, state_dim)
                              or (state_dim,).
                              
        Returns:
            torch.Tensor: Predicted Q-values of shape (batch_size, action_dim)
                          or (action_dim,).
        """
        return self.net(x)


# -----------------------------------------------------------------------------
# Experience Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    """
    Experience Replay Buffer for Deep Q-Learning.
    
    Stores past transition tuples (state, action, reward, next_state, done).
    Sampling random mini-batches from this buffer breaks the temporal correlations
    inherent in sequential reinforcement learning, stabilizing gradient descent.
    """
    def __init__(self, capacity=20000):
        """
        Initializes a circular FIFO buffer.
        
        Args:
            capacity (int): Maximum number of transition tuples to retain in memory.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state_vec, action, reward, next_state_vec, done):
        """
        Appends a transition tuple into the replay buffer.
        
        Args:
            state_vec (torch.Tensor): Feature tensor of the starting state s.
            action (int): Discrete action index taken (0..3).
            reward (float): Reward received from environment after transition.
            next_state_vec (torch.Tensor): Feature tensor of the resulting state s'.
            done (bool): Whether the episode terminated (crashed or parked).
        """
        self.buffer.append((state_vec, action, reward, next_state_vec, done))

    def sample(self, batch_size):
        """
        Draws a uniform random batch of transitions and packs them into PyTorch tensors.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) as stacked PyTorch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        """Returns the current number of experiences stored in the buffer."""
        return len(self.buffer)


# -----------------------------------------------------------------------------
# Deep Q-Learning Agent (DQN)
# -----------------------------------------------------------------------------
class DQNAgent:
    """
    Deep Q-Network (DQN) Agent.
    
    Key Features:
    - Dual Networks: Online Policy Network (θ) and Target Network (θ_target)
    - Soft Target Updates: Polyak averaging with factor τ = 0.005
    - Huber Loss (SmoothL1Loss) and Gradient Clipping for training stability
    - Feature Normalization: Encodes coordinates, target vector, one-hot heading, and LIDAR
    - Seamless ASCII Visualizer Integration: Exposes get_q() and select_action()
    """
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.98,
        epsilon=1.0,
        eps_min=0.03,
        eps_decay=0.995,
        tau=0.005,
        batch_size=64,
        device=None,
    ):
        """
        Initializes the DQN Agent and its hyperparameters.
        
        Args:
            env (ParkingEnvironment): The simulation parking environment.
            lr (float): Learning rate for the Adam optimizer (default: 1e-3).
            gamma (float): Discount factor for future rewards γ ∈ [0, 1] (default: 0.98).
            epsilon (float): Initial exploration rate for ε-greedy policy (default: 1.0).
            eps_min (float): Minimum floor for exploration rate (default: 0.03).
            eps_decay (float): Multiplicative decay applied per episode (default: 0.995).
            tau (float): Polyak soft-update rate for target network (default: 0.005).
            batch_size (int): Mini-batch size sampled during training steps (default: 64).
            device (torch.device or str, optional): Hardware device ('cuda' or 'cpu').
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.tau = tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim=12, action_dim=4).to(self.device)
        self.target_net = QNetwork(state_dim=12, action_dim=4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=20000)

    def get_state_vector(self, state):
        """
        Encodes car position, orientation, target vector, and LIDAR into a 12-dim float tensor.
        
        Feature Breakdown:
        - [0..1]:  Normalized coordinates (x / (W-1), y / (H-1))
        - [2..3]:  Normalized vector towards target parking bay (dx / W, dy / H)
        - [4..7]:  One-hot vehicle heading [North, East, South, West]
        - [8..11]: 4-directional LIDAR obstacle distance readings [Front, Rear, Left, Right]
        
        Args:
            state (tuple): (x, y, orientation) where orientation ∈ {0, 1, 2, 3}.
            
        Returns:
            torch.Tensor: 1D float tensor of shape (12,) on the agent's device.
        """
        x, y, o = state
        lidar = self.env.get_lidar(x, y, o)

        nx = x / float(self.env.width - 1)
        ny = y / float(self.env.height - 1)

        tx, ty = self.env.get_closest_target(x, y)
        dx = (tx - x) / float(self.env.width)
        dy = (ty - y) / float(self.env.height)

        orientations = [0.0, 0.0, 0.0, 0.0]
        orientations[o] = 1.0

        lidar_readings = [
            min(lidar["front"], 16) / 16.0,
            min(lidar["rear"], 16) / 16.0,
            min(lidar["left"], 16) / 16.0,
            min(lidar["right"], 16) / 16.0,
        ]

        features = [nx, ny, dx, dy] + orientations + lidar_readings
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def get_q(self, state):
        """
        Computes the Q-values for all 4 actions from the neural network.
        
        Used by the AsciiVisualizer to draw the live confidence gauges on the HUD.
        
        Args:
            state (tuple): (x, y, orientation).
            
        Returns:
            list[float]: 4 action Q-values [Q_FWD, Q_REV, Q_LEFT, Q_RIGHT].
        """
        self.policy_net.eval()
        with torch.no_grad():
            s_vec = self.get_state_vector(state)
            q_vals = self.policy_net(s_vec).cpu().numpy().tolist()
        return q_vals

    def select_action(self, state, explore=True):
        """
        Selects an action using an epsilon-greedy policy:
        - With probability ε (if explore=True): Pick a random uniform action (Exploration).
        - With probability 1 - ε: Pick action with highest Q-value from policy net (Exploitation).
        
        Args:
            state (tuple): (x, y, orientation).
            explore (bool): Whether to allow random exploratory moves.
            
        Returns:
            int: Action index (0: FWD, 1: REV, 2: TURN_LEFT, 3: TURN_RIGHT).
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, 3)

        self.policy_net.eval()
        with torch.no_grad():
            s_vec = self.get_state_vector(state)
            q_vals = self.policy_net(s_vec)
            return int(q_vals.argmax().item())

    def train_step(self):
        """
        Executes one gradient descent optimization step on a mini-batch from replay memory.
        
        Algorithm:
        1. Sample mini-batch of (s, a, r, s', done) from ReplayBuffer.
        2. Compute current Q-values: Q_policy(s, a).
        3. Compute Bellman target:
               y = r + γ * (1 - done) * max_a' Q_target(s', a')
        4. Calculate Huber loss: L = SmoothL1(Q_policy(s, a), y).
        5. Backpropagate and clip gradients to max norm 10.0.
        6. Polyak soft-update target network parameters:
               θ_target ← τ * θ_policy + (1 - τ) * θ_target
               
        Returns:
            float or None: Loss value for this step, or None if buffer has insufficient samples.
        """
        if len(self.memory) < self.batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak soft-update target network: θ_target = τ * θ_policy + (1 - τ) * θ_target
        for target_p, policy_p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_p.data.copy_(self.tau * policy_p.data + (1.0 - self.tau) * target_p.data)

        return loss.item()

    def decay_epsilon(self):
        """Decays exploration factor ε after each episode: ε = max(ε_min, ε * ε_decay)."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path="autopark_dqn.pt"):
        """
        Saves model weights, target network, optimizer state, and epsilon to disk.
        
        Args:
            path (str): File path for the saved PyTorch checkpoint.
        """
        torch.save({
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path="autopark_dqn.pt"):
        """
        Loads saved model weights from disk.
        
        Args:
            path (str): File path to the PyTorch checkpoint.
            
        Returns:
            bool: True if loaded successfully, False if file not found.
        """
        if not os.path.isfile(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
        self.epsilon = checkpoint.get("epsilon", self.eps_min)
        return True


# -----------------------------------------------------------------------------
# Training Engine
# -----------------------------------------------------------------------------
def train_dqn(env, agent, episodes=800, max_steps=60, verbose=True):
    """
    Runs the Deep Q-Network training loop across multiple parking episodes.
    
    Training Cycle:
    1. Spawn car (alternates 50% from entrance gate, 50% from random open aisle/bay).
    2. At each step:
       - Query agent for action (ε-greedy).
       - Step the physics environment to obtain next state and reward.
       - Store transition (s, a, r, s', done) in the ReplayBuffer.
       - Execute a mini-batch gradient descent update on the policy network.
    3. Decay exploration rate ε at the end of each episode.
    4. Track rolling 100-episode success rate and display a terminal progress bar.
    
    Args:
        env (ParkingEnvironment): The parking lot grid simulation environment.
        agent (DQNAgent): The neural network agent to train.
        episodes (int): Total number of training episodes (default: 800).
        max_steps (int): Maximum actions allowed per episode before timeout (default: 60).
        verbose (bool): Whether to stream progress bars and statistics to stdout.
        
    Returns:
        DQNAgent: The trained agent with optimized neural network weights.
    """
    start_time = time.time()
    recent_history = []

    if verbose:
        print("\n" + Color.CYAN + "═" * 66 + Color.RESET)
        print(f" {Color.BOLD}🧠 TRAINING DEEP Q-NETWORK (DQN) WITH PYTORCH ({episodes:,} EPISODES)...{Color.RESET}")
        print(f"    Device: {agent.device} | Architecture: MLP 12 -> 128 -> 128 -> 4")
        print(Color.CYAN + "═" * 66 + Color.RESET)

    for ep in range(1, episodes + 1):
        if ep % 2 == 0:
            sx, sy = env.start
            so = 1
        else:
            sx, sy = random.choice(env.valid_spawns)
            so = random.randint(0, 3)

        state = (sx, sy, so)
        parked = False

        for _ in range(max_steps):
            s_vec = agent.get_state_vector(state)
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(state, action)
            next_s_vec = agent.get_state_vector(next_state)

            agent.memory.push(s_vec, action, reward, next_s_vec, done)
            agent.train_step()

            state = next_state
            if done:
                parked = (info == "PARKED")
                break

        recent_history.append(1 if parked else 0)
        if len(recent_history) > 100:
            recent_history.pop(0)

        agent.decay_epsilon()

        if verbose and (ep % max(1, episodes // 20) == 0 or ep == episodes):
            rate = (sum(recent_history) / len(recent_history)) * 100 if recent_history else 0
            bar_len = 24
            filled = int(bar_len * (ep / episodes))
            pbar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r Progress: [{pbar}] {ep:4d}/{episodes} ep | "
                f"Success Rate: {rate:5.1f}% | "
                f"Eps: {agent.epsilon:.2f} | "
                f"Time: {elapsed:.2f}s "
            )
            sys.stdout.flush()

    if verbose:
        print("\n" + Color.GREEN + " ✔ PyTorch DQN Training Complete!" + Color.RESET)
        time.sleep(0.5)

    return agent


# -----------------------------------------------------------------------------
# Main CLI & Interactive Driver
# -----------------------------------------------------------------------------
def main():
    """
    Main entry point and CLI driver for the PyTorch Deep Q-Network simulation.
    
    Workflow:
    1. Parse command-line arguments (--episodes, --speed, --no-color, --load, --model-path).
    2. Initialize ParkingEnvironment, AsciiVisualizer, and DQNAgent.
    3. Either train a fresh model or load pre-trained weights from disk.
    4. Run an autonomous animated parking demonstration using AsciiVisualizer.
    5. Present an interactive post-run menu allowing the user to test random spawns
       or repeat entrance parking.
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Self-Parking Car with PyTorch Deep Q-Network (DQN)"
    )
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes (default: 800)")
    parser.add_argument("--speed", type=float, default=0.20, help="Animation step delay in seconds (default: 0.20)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--map", type=str, default="default_map.txt", help="Path to custom ASCII map file (default: default_map.txt)")
    parser.add_argument("--model-path", type=str, default="autopark_dqn.pt", help="Path to save/load weights")
    parser.add_argument("--load", action="store_true", help="Load saved model without re-training")
    args = parser.parse_args()

    use_color = not args.no_color
    env = ParkingEnvironment(map_file=args.map)
    visualizer = AsciiVisualizer(env, use_color=use_color)
    agent = DQNAgent(env)

    print_banner()

    if args.load:
        if agent.load(args.model_path):
            print(Color.GREEN + f" ✔ Loaded PyTorch model weights from '{args.model_path}'" + Color.RESET)
        else:
            print(Color.RED + f" [!] Could not find '{args.model_path}'. Training a new model..." + Color.RESET)
            train_dqn(env, agent, episodes=args.episodes, verbose=True)
            agent.save(args.model_path)
    else:
        train_dqn(env, agent, episodes=args.episodes, verbose=True)
        agent.save(args.model_path)
        print(Color.GREEN + f" ✔ Model weights saved to '{args.model_path}'" + Color.RESET)

    # Launch demonstration
    print("\n Launching PyTorch Neural Network Autonomous Parking Demo...\n")
    time.sleep(0.8)
    run_animated_episode(env, agent, visualizer, step_delay=args.speed, title="PyTorch DQN Policy")

    # Post-run prompt
    print("\n" + Color.GREEN + Color.BOLD + "🎉 PYTORCH NEURAL NETWORK PARKED SUCCESSFULLY! 🎉" + Color.RESET)
    print("\nWhat would you like to try next?")
    print("  1. Run 5 Random Spawn Challenges")
    print("  2. Run Another Demo from Entrance")
    print("  3. Exit")

    try:
        choice = input("\nSelect an option [1-3] (default 1): ").strip()
        if choice == "2":
            run_animated_episode(env, agent, visualizer, step_delay=args.speed, title="PyTorch Demo")
        elif choice == "3":
            print("Exiting. Goodbye!")
        else:
            multi_spot_challenge(env, agent, visualizer, num_tests=5)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n")
        print("\nSimulation aborted by user.")
