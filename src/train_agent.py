import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from env_sp500 import SP500TradingEnv

# === Set the data file here ===
DATA_FILE = 'forex.csv'  # Change to 'sp500.csv' for S&P 500


def main():
    # Load data, skip extra header rows
    df = pd.read_csv(DATA_FILE, skiprows=2)
    df = df.rename(columns={'Unnamed: 1': 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    # Create vectorized environment with Monitor
    def env_fn():
        return SP500TradingEnv(df, max_steps_per_episode=100)
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    env = make_vec_env(env_fn, n_envs=1, monitor_dir=log_dir)
    # Train agent with more exploration
    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    model.learn(total_timesteps=100000)
    model.save("sp500_dqn_agent")
    # Plot results
    suffix = 'sp500' if 'sp500' in DATA_FILE else 'forex'
    plot_results(log_dir, suffix)
    plot_final_profits(log_dir, suffix)

def plot_results(log_dir, suffix):
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'Training Rewards ({suffix.upper()})')
    plt.savefig(f'training_rewards_{suffix}.png')
    plt.close()

def plot_final_profits(log_dir, suffix):
    # Find the correct monitor file
    monitor_files = [f for f in os.listdir(log_dir) if f.endswith('.monitor.csv')]
    if not monitor_files:
        print(f"No monitor file found in {log_dir}.")
        return
    monitor_file = os.path.join(log_dir, monitor_files[0])
    df = pd.read_csv(monitor_file, skiprows=1)  # skip header
    if 'r' not in df.columns:
        print(f"Column 'r' not found in monitor file {monitor_file}.")
        return
    plt.plot(df['r'])
    plt.xlabel('Episode')
    plt.ylabel('Final Profit')
    plt.title(f'Final Profits per Episode ({suffix.upper()})')
    plt.savefig(f'final_profits_{suffix}.png')
    plt.close()

if __name__ == '__main__':
    main() 