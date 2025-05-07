import pandas as pd
from stable_baselines3 import DQN
from env_sp500 import SP500TradingEnv
import matplotlib.pyplot as plt

# === Set the data file here ===
DATA_FILE = 'forex.csv'  # Change to 'sp500.csv' for S&P 500

def evaluate_agent(model_path, data_path, window_size=10, initial_balance=10000):
    df = pd.read_csv(data_path, skiprows=2)
    df = df.rename(columns={'Unnamed: 1': 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    env = SP500TradingEnv(df, window_size=window_size, initial_balance=initial_balance)
    obs, _ = env.reset()
    done = False
    balances = [env.balance]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        balances.append(env.balance + env.shares_held * df['Close'].iloc[env.current_step-1])
    return balances, env.total_profit, env.trades

if __name__ == '__main__':
    model = DQN.load('sp500_dqn_agent')
    balances, total_profit, trades = evaluate_agent('sp500_dqn_agent', DATA_FILE)
    suffix = 'sp500' if 'sp500' in DATA_FILE else 'forex'
    filename = f'equity_curve_{suffix}.png'
    print(f"Saving equity curve to {filename} (suffix: {suffix})")
    plt.plot(balances)
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.title(f'Equity Curve ({suffix.upper()})')
    plt.savefig(filename)
    plt.show()
    print(f'Total Profit: ${total_profit:.2f}')
    print(f'Total Trades: {trades}') 