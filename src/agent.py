import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Multi-layer Perceptron (MLP) mapping state to action-values (Q-values).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Experience Replay Buffer to store and sample transitions.
    Reduces correlation between consecutive transitions.
    """
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent.
    """
    def __init__(self, state_dim=3, action_dim=7, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, buffer_capacity=20000, target_update_freq=10,
                 cql_alpha=0.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.cql_alpha = cql_alpha
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Policy network & Target network
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
        self.steps_done = 0
        self.update_count = 0

    def _normalize_state(self, state):
        """
        Normalizes the state variables for the original Duolingo dataset features:
        - state[0]: history_seen -> scale by dividing by 100.0 (clip at 1.0)
        - state[1]: history_correct -> scale by dividing by 100.0 (clip at 1.0)
        - state[2]: delta_days -> scale to [0, 1] by dividing by 365.0
        """
        state_np = np.array(state, dtype=np.float32).copy()
        
        # Clip max values to prevent outliers from distorting the network
        if len(state_np.shape) == 1:
            state_np[0] = min(state_np[0] / 100.0, 1.0)
            state_np[1] = min(state_np[1] / 100.0, 1.0)
            state_np[2] = min(state_np[2] / 365.0, 1.0)
        else:
            state_np[:, 0] = np.clip(state_np[:, 0] / 100.0, 0.0, 1.0)
            state_np[:, 1] = np.clip(state_np[:, 1] / 100.0, 0.0, 1.0)
            state_np[:, 2] = np.clip(state_np[:, 2] / 365.0, 0.0, 1.0)
            
        return state_np

    def select_action(self, state, evaluate=False):
        """
        Selects an action using epsilon-greedy policy.
        """
        norm_state = self._normalize_state(state)
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(norm_state).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        norm_state = self._normalize_state(state)
        norm_next_state = self._normalize_state(next_state)
        self.memory.add(norm_state, action, reward, norm_next_state, done)

    def update(self):
        """
        Samples a batch and performs one gradient descent step.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CQL-lite: penalize Q-values of out-of-distribution actions relative to 
        # the action actually taken in the logged data (Kumar et al. 2020)
        logsumexp_q = torch.logsumexp(q_values, dim=1)
        cql_penalty = (logsumexp_q - state_action_values).mean()
        
        # Double DQN: select best action using Policy Net, evaluate using Target Net
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_state_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            # y_i = r + gamma * Q_target(s', argmax Q_policy(s', a')) * (1 - done)
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1.0 - dones))
        # Compute Loss (Huber Loss instead of MSE)
        bellman_loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
        loss = bellman_loss + self.cql_alpha * cql_penalty
        
        self.last_max_q = q_values.abs().max().item()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients and log for the first 5 steps
        total_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        if getattr(self, 'update_count', 0) < 5:
            clipped_norm = sum(p.grad.data.norm(2).item()**2 for p in self.policy_net.parameters() if p.grad is not None)**0.5
            print(f"[Grad Check] Step {self.update_count+1}: Norm before clip={total_norm:.4f}, after clip={clipped_norm:.4f}")

        self.optimizer.step()
        
        self.update_count += 1
        
        # Soft or periodic update of target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)
            
        return loss.item()

    def save(self, filepath):
        torch.save({
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
