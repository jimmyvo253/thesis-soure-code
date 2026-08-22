import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import time
from env import FlashcardEnv
from agent import DQNAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# ONLINE TRAINING CODE
# =============================================================================
def train_dqn(num_episodes=800, max_steps_per_episode=30, save_path="models/dqn_agent_online.pt"):
    start_time = time.time()
    set_seed(42)
    # Create target directory for models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize env & agent
    env = FlashcardEnv(max_steps=max_steps_per_episode, target_recall=0.85, seed=42)
    agent = DQNAgent(state_dim=3, action_dim=7, lr=5e-4, gamma=0.95, 
                     epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.992,
                     cql_alpha=0.0)
    
    rewards_history = []
    moving_avg_rewards = []
    loss_history = []
    
    print("Starting Online DQN Agent training...")
    print(f"Training parameters: episodes={num_episodes}, max_steps={max_steps_per_episode}")
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, float(done))
            loss = agent.update()
            
            if loss is not None:
                loss_history.append(loss)
                
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        
        # Calculate moving average (window=30)
        if len(rewards_history) >= 30:
            avg_rew = np.mean(rewards_history[-30:])
        else:
            avg_rew = np.mean(rewards_history)
        moving_avg_rewards.append(avg_rew)
        
        if episode % 500 == 0:
            avg_loss = np.mean(loss_history[-100:]) if len(loss_history) > 0 else 0.0
            max_q = getattr(agent, 'last_max_q', 0.0)
            print(f"Episode {episode:03d}/{num_episodes} | Avg Reward (last 30): {avg_rew:7.2f} | Loss: {avg_loss:6.4f} | Epsilon: {agent.epsilon:5.3f} | Max |Q|: {max_q:.4f}")
            
    # Save the trained model
    agent.save(save_path)
    print(f"Online Model saved to {save_path}")
    
    # Save rewards_history to CSV
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"episode": range(1, len(rewards_history) + 1), "reward": rewards_history}).to_csv("results/online_reward_log.csv", index=False)
    print("rewards_history saved to results/online_reward_log.csv")
    
    # Generate and save convergence plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Episode Reward", alpha=0.3, color="blue")
    plt.plot(moving_avg_rewards, label="Moving Avg Reward (last 30)", color="darkblue", linewidth=2)
    plt.title("Online DQN Learning Curve (Flashcard Spaced Repetition)")
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save the plot
    plot_path = "results/dqn_learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curve plot saved to {plot_path}")
    
    duration = time.time() - start_time
    print(f"Online training execution time: {duration:.2f} seconds")
    
    return agent

# =============================================================================
# OFFLINE TRAINING CODE
# =============================================================================
def train_dqn_offline(data_path="processed/anki_processed.csv", save_path="models/dqn_agent_offline.pt", transition_cap=100000):
    start_time = time.time()
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dataset_fingerprint import check_processed_dataset
    check_processed_dataset(data_path)
    
    set_seed(42)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize agent with buffer capacity equal to or greater than the transition cap
    buf_capacity = max(200000, transition_cap)
    agent = DQNAgent(state_dim=3, action_dim=7, lr=1e-4, gamma=0.95, 
                     epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.999,
                     buffer_capacity=buf_capacity, target_update_freq=200, cql_alpha=1.0) # Slower decay for batch learning
    
    print("Starting Offline DQN Agent training (Anki Dataset)...")
    print(f"Loading data from {data_path}...")
    print(f"Parameters: transition_cap={transition_cap}, buffer_capacity={buf_capacity}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run preprocess_anki_10k.py first.")
        
    df = pd.read_csv(data_path)
    
    # Sort by group_id and day_offset (Việc 3)
    df.sort_values(by=['group_id', 'day_offset'], inplace=True)
    
    print("Data loaded. Extracting transitions...")
    
    INTERVALS = np.array([1, 2, 4, 7, 15, 30, 60])
    
    # For a given group_id (representing a user-card sequence), transition from row i to row i+1 (Việc 3)
    df['next_group_id'] = df['group_id'].shift(-1)
    
    # Identify valid transitions (same group)
    valid_transitions = (df['group_id'] == df['next_group_id'])
    
    # Shift necessary columns for next state and reward using new Anki schema (Việc 3)
    df['next_history_seen'] = df['cumulative_reviews'].shift(-1)
    df['next_history_correct'] = df['cumulative_correct'].shift(-1)
    df['next_delta_t'] = df['delta_t'].shift(-1)
    df['next_y'] = df['y'].shift(-1)
    
    # Filter valid rows
    tdf = df[valid_transitions].copy()
    tdf = tdf.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Found {len(tdf)} valid transitions. Populating Replay Buffer...")
    
    # Populate Replay Buffer
    count = 0
    for _, row in tdf.iterrows():
        # Action is the actual time elapsed until the next review
        action = int(np.argmin(np.abs(INTERVALS - row['next_delta_t'])))
        
        state = np.array([row['cumulative_reviews'], row['cumulative_correct'], row['delta_t']])
        next_state = np.array([row['next_history_seen'], row['next_history_correct'], row['next_delta_t']])
        
        # Reward based on the outcome of the NEXT review (Việc 4: keep binary reward)
        reward = 1.0 if row['next_y'] == 1 else -1.0
        done = 0.0
        
        agent.store_transition(state, action, reward, next_state, done)
        count += 1
        
        # Cap buffer size
        if count >= transition_cap:
            break
            
    print(f"Buffer populated with {len(agent.memory)} transitions.")
    
    # Batch Training
    num_updates = 6000
    loss_history = []
    
    print(f"Starting {num_updates} offline batch updates...")
    for step in range(1, num_updates + 1):
        loss = agent.update()
        if loss is not None:
            loss_history.append(loss)
            
        if step % 500 == 0:
            avg_loss = np.mean(loss_history[-100:]) if len(loss_history) > 0 else 0.0
            max_q = getattr(agent, 'last_max_q', 0.0)
            print(f"Update Step {step:04d}/{num_updates} | Loss: {avg_loss:6.4f} | Epsilon: {agent.epsilon:5.3f} | Max |Q|: {max_q:.4f}")
            
    # Save the trained model
    agent.save(save_path)
    print(f"Offline Model saved to {save_path}")
    
    # Save loss_history to CSV
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"step": range(1, len(loss_history) + 1), "loss": loss_history}).to_csv("results/offline_loss_log.csv", index=False)
    print("loss_history saved to results/offline_loss_log.csv")
    
    # Generate and save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss", alpha=0.3, color="red")
    
    # Moving average
    if len(loss_history) > 50:
        smoothed_loss = pd.Series(loss_history).rolling(window=50).mean()
        plt.plot(smoothed_loss, label="Smoothed Loss (last 50)", color="darkred", linewidth=2)
        
    plt.title("Offline DQN Training Loss (Anki Dataset)")
    plt.xlabel("Update Steps")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plot_path = "results/dqn_offline_loss.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")
    
    duration = time.time() - start_time
    print(f"Offline training execution time: {duration:.2f} seconds")
    
    return agent

if __name__ == "__main__":
    # Train online DQN with 3000 episodes
    train_dqn(num_episodes=3000, save_path="models/_temp_for_logging_only.pt")
    
    # Train offline DQN with official transition cap = 100,000
    train_dqn_offline(save_path="models/_temp_for_logging_only.pt", transition_cap=100000)
