import numpy as np

class UserSimulator:
    """
    Simulates a human learner's memory forgetting and retention behaviors.
    Based on the Exponential Forgetting Curve: P(recall) = exp(-t / h)
    where:
      - t: time elapsed since the last review (in days)
      - h: memory stability/half-life (in days)
      - d: card difficulty (0.1 to 1.0, where 1.0 is extremely hard)
    """
    def __init__(self, num_cards=100, seed=42):
        self.num_cards = num_cards
        self.np_random = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        # Initialize card difficulties: uniform between 0.1 (very easy) and 0.9 (very hard)
        self.difficulties = self.np_random.uniform(0.1, 0.9, size=self.num_cards)
        
        # Initialize memory half-life for each card: easy cards start with higher stability
        # h_init ranges from 2.0 days (for hard cards) to 8.0 days (for easy cards)
        self.half_lives = 8.0 - 6.0 * self.difficulties
        
        # Track when each card was last reviewed (virtual time, starts at 0.0)
        self.last_reviewed = np.zeros(self.num_cards)
        
        # Virtual current time in days
        self.current_time = 0.0
        
        # Study history logs for evaluation
        # Format: (card_id, timestamp, interval, recalled, old_half_life, new_half_life)
        self.history = []
        
        # Tracking history counters for DQN state representation
        self.history_seen = np.zeros(self.num_cards, dtype=np.float32)
        self.history_correct = np.zeros(self.num_cards, dtype=np.float32)
        
        return self.get_state_all()

    def get_recall_probability(self, card_id, current_time):
        t = current_time - self.last_reviewed[card_id]
        h = self.half_lives[card_id]
        # P(recall) = 2^(-t / h)
        return 2.0 ** (-t / h)

    def review_card(self, card_id, interval):
        """
        Simulates the review of a card after a certain interval (in days).
        Updates the virtual current time and the card's memory stability (half-life).
        """
        # Update current time based on when the review actually happens
        # In simulator, we assume the user reviews the card at: last_reviewed + interval
        review_time = self.last_reviewed[card_id] + interval
        if review_time > self.current_time:
            self.current_time = review_time
            
        t = interval
        h_old = self.half_lives[card_id]
        d = self.difficulties[card_id]
        
        # Compute probability of recall before the review
        # P(recall) = 2^(-t / h_old)
        p_recall = 2.0 ** (-t / h_old)
        
        # Determine if user successfully recalled the card (stochastic)
        recalled = self.np_random.rand() < p_recall
        
        # Update half-life based on outcome (Spaced Repetition Theory)
        if recalled:
            # Spacing effect: recalling with lower P(recall) leads to larger half-life increase.
            # Difficulty factor: easy cards scale up faster than hard cards.
            difficulty_factor = 1.3 - 0.8 * d  # multiplier between 0.5 and 1.22
            factor = (1.5 + 4.0 * (1.0 - p_recall)) * difficulty_factor
            # Clamp the multiplier to prevent exponential explosion or negative scaling
            factor = max(1.1, min(factor, 8.0))
            h_new = h_old * factor
        else:
            # If forgotten, reset half-life to a lower value (penalty for forgetting)
            h_new = max(0.5, h_old * 0.3)
            
        self.half_lives[card_id] = h_new
        self.last_reviewed[card_id] = self.current_time
        
        # Update history counters for state representation
        self.history_seen[card_id] += 1.0
        if recalled:
            self.history_correct[card_id] += 1.0
            
        # Log the review event
        self.history.append({
            'card_id': int(card_id),
            'time': float(self.current_time),
            'interval': float(t),
            'p_recall': float(p_recall),
            'recalled': bool(recalled),
            'old_h': float(h_old),
            'new_h': float(h_new),
            'difficulty': float(d)
        })
        
        return recalled, h_new

    def get_state(self, card_id):
        # State of a single card: [history_seen, history_correct, days_since_last_review]
        t = self.current_time - self.last_reviewed[card_id]
        return np.array([
            self.history_seen[card_id],
            self.history_correct[card_id],
            t
        ], dtype=np.float32)

    def get_state_all(self):
        states = []
        for i in range(self.num_cards):
            states.append(self.get_state(i))
        return np.array(states, dtype=np.float32)
