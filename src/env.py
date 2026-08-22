import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FlashcardEnv(gym.Env):
    # state: [history_seen, history_correct, t]
    # actions: 7 intervals
    metadata = {"render_modes": ["human"]}
    
    INTERVALS = [1, 2, 4, 7, 15, 30, 60]

    def __init__(self, max_steps=30, target_recall=0.85, seed=None):
        super().__init__()
        self.max_steps = max_steps
        self.target_recall = target_recall
        
        # State: [history_seen, history_correct, time_since_last_review]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 365.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 7 discrete intervals
        self.action_space = spaces.Discrete(len(self.INTERVALS))
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Initialize difficulty: [0.1, 0.9]
        self.difficulty = self.np_random.uniform(0.1, 0.9)
        
        # Initial half-life: easy cards start stronger (up to 8.0 days), hard cards weaker (2.0 days)
        self.half_life = 8.0 - 6.0 * self.difficulty
        
        # Time since last review starts at 0.0
        self.t_since_last = 0.0
        
        self.steps = 0
        
        # Tracking history counters for state representation
        self.history_seen = 0.0
        self.history_correct = 0.0
        
        state = np.array([self.history_seen, self.history_correct, self.t_since_last], dtype=np.float32)
        info = {}
        return state, info

    def step(self, action):
        interval = self.INTERVALS[action]
        
        # P(recall) formula
        p_recall = 2.0 ** (-interval / self.half_life)
        
        # random roll
        recalled = self.np_random.rand() < p_recall
        
        h_old = self.half_life
        
        if recalled:
            # spacing factor
            difficulty_factor = 1.3 - 0.8 * self.difficulty
            factor = (1.5 + 4.0 * (1.0 - p_recall)) * difficulty_factor
            factor = max(1.1, min(factor, 8.0))
            self.half_life = h_old * factor
            
            # shape reward
            reward = 1.0 - 2.0 * abs(p_recall - self.target_recall)
        else:
            # penalize forgetting
            self.half_life = max(0.5, h_old * 0.3)
            reward = -1.0
            
        # Clamp half-life to prevent infinite growth
        self.half_life = min(self.half_life, 365.0)
        
        # Update history counters
        self.history_seen += 1.0
        if recalled:
            self.history_correct += 1.0
            
        # The time elapsed in the next state is the action interval we just simulated
        self.t_since_last = float(interval)
        self.steps += 1
        
        terminated = self.steps >= self.max_steps
        truncated = False
        
        next_state = np.array([self.history_seen, self.history_correct, self.t_since_last], dtype=np.float32)
        
        info = {
            "recalled": recalled,
            "p_recall": p_recall,
            "old_half_life": h_old,
            "new_half_life": self.half_life,
            "interval": interval
        }
        
        return next_state, reward, terminated, truncated, info
