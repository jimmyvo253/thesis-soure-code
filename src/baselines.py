import numpy as np

class RandomScheduler:
    """
    Randomly chooses review intervals.
    """
    def __init__(self, action_dim=7):
        self.action_dim = action_dim

    # def select_action(self, state):
    #     return np.random.randint(0, self.action_dim)

    def select_action(self, state, rng=None):
        if rng is not None:
            return rng.randint(0, self.action_dim)
        return np.random.randint(0, self.action_dim)


class LeitnerScheduler:
    """
    Classic Leitner box system scheduler.
    - If recalled correctly: card moves to the next box (longer interval).
    - If forgotten: card resets to box 0 (1 day).
    """
    def __init__(self, intervals=[1, 2, 4, 7, 15, 30, 60]):
        self.intervals = intervals
        self.box = 0

    def reset(self):
        self.box = 0

    def select_action(self, recalled=True):
        if recalled:
            self.box = min(self.box + 1, len(self.intervals) - 1)
        else:
            self.box = 0
        return self.box


class SM2Scheduler:
    """
    SuperMemo-2 (SM-2) algorithm.
    Tracks repetitions (n) and Easiness Factor (EF).
    """
    def __init__(self, intervals=[1, 2, 4, 7, 15, 30, 60]):
        self.intervals = intervals
        self.n = 0          # Number of consecutive correct repetitions
        self.ef = 2.5       # Easiness factor (starts at 2.5)
        self.interval = 1.0 # Current interval in days

    def reset(self):
        self.n = 0
        self.ef = 2.5
        self.interval = 1.0

    def select_action(self, recalled, p_recall):
        """
        Calculates next interval based on SM-2 rules, maps to the closest action index.
        """
        # Convert binary recall + probability of recall into SM-2 grade (0 to 5)
        if recalled:
            if p_recall > 0.85:
                grade = 5  # Perfect response
            elif p_recall > 0.60:
                grade = 4  # Correct response after hesitation
            else:
                grade = 3  # Correct response with serious difficulty
        else:
            if p_recall > 0.40:
                grade = 2  # Incorrect, but easy to recall
            elif p_recall > 0.20:
                grade = 1  # Incorrect, remembered with effort
            else:
                grade = 0  # Complete blackout
                
        # Update repetition count and interval
        if grade >= 3:
            if self.n == 0:
                self.interval = 1.0
            elif self.n == 1:
                self.interval = 4.0  # SM-2 uses 6.0 generally, we use 4.0 to align closer to our action space
            else:
                self.interval = self.interval * self.ef
            self.n += 1
        else:
            self.n = 0
            self.interval = 1.0

        # Update Easiness Factor (EF)
        self.ef = self.ef + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
        self.ef = max(1.3, self.ef)
        
        # Map self.interval (continuous days) to closest discrete action in INTERVALS
        # Find action index with minimum absolute difference
        action = int(np.argmin([abs(self.interval - i) for i in self.intervals]))
        return action


class HLRScheduler:
    """
    Half-Life Regression (HLR) Scheduler.
    Schedules reviews based on predicted memory half-life:
    h = 2^(theta^T x)
    and maps the optimal interval to the closest action index.
    """
    def __init__(self, weights_path, intervals=[1, 2, 4, 7, 15, 30, 60], target_recall=0.85):
        self.intervals = intervals
        self.target_recall = target_recall
        
        # Load weights from JSON
        import json
        with open(weights_path, 'r') as f:
            data = json.load(f)
        self.theta = np.array(data['theta'])
        self.model_type = data.get('model_type', 'original')

    def select_action(self, state):
        # state is [history_seen, history_correct, days_since_last_review]
        hs = state[0]
        hc = state[1]
        hw = hs - hc
        
        if self.model_type == 'original':
            x = np.array([
                1.0, 
                np.log2(1.0 + hc), 
                np.log2(1.0 + hw)
            ])
        else: # extended
            acc = hc / hs if hs > 0 else 1.0
            # review_count is hs + 1
            rev_cnt = hs + 1.0
            x = np.array([
                1.0, 
                np.log2(1.0 + hc), 
                np.log2(1.0 + hw), 
                acc, 
                np.log2(rev_cnt)
            ])
            
        # h = 2^(theta^T x)
        h = 2.0 ** np.dot(self.theta, x)
        h = np.clip(h, 0.01, 36500.0)
        
        # t = -h * log2(P_target)
        t_opt = -h * np.log2(self.target_recall)
        
        # Map to closest action index in self.intervals
        action = int(np.argmin([abs(t_opt - i) for i in self.intervals]))
        return action

