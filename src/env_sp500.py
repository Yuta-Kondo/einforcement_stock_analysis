import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import csv
import os

class SP500TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=10, initial_balance=10000, log_actions_path='actions_log.csv', transaction_cost=0.0, max_steps_per_episode=100):
        super(SP500TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 2,), dtype=np.float32
        )
        self.log_actions_path = log_actions_path
        self.transaction_cost = transaction_cost
        self.max_steps_per_episode = max_steps_per_episode
        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.total_profit = 0
        self.trades = 0
        self.prev_buy_price = None
        self.actions = []
        self.episode_steps = 0
        # Force initial buy
        price = self.df['Close'].iloc[self.current_step]
        if self.balance >= price:
            self.shares_held = 1
            self.balance -= price
            self.prev_buy_price = price
            self.trades += 1
            self.actions.append((self.current_step, 1, self.balance, self.shares_held, price))
            print(f"[RESET] Forced initial buy at step {self.current_step}: balance={self.balance}, shares_held={self.shares_held}, price={price}")
        else:
            print(f"[RESET] Could not force initial buy at step {self.current_step}: balance={self.balance}, price={price}")
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df['Close'].iloc[self.current_step - self.window_size:self.current_step].values
        obs = np.concatenate([
            window / window[0] - 1,  # normalize window
            [self.balance / self.initial_balance - 1],
            [self.shares_held]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        price = self.df['Close'].iloc[self.current_step]
        reward = 0
        # Log action
        self.actions.append((self.current_step, action, self.balance, self.shares_held, price))
        # Execute action
        if action == 1:  # Buy
            if self.balance >= price and self.shares_held == 0:
                self.shares_held = 1
                self.balance -= price
                self.prev_buy_price = price
                self.trades += 1
                print(f"[STEP] Buy at step {self.current_step}: balance={self.balance}, shares_held={self.shares_held}, price={price}")
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held = 0
                self.balance += price
                reward = price - self.prev_buy_price if self.prev_buy_price is not None else 0
                self.prev_buy_price = None
                self.trades += 1
                print(f"[STEP] Sell at step {self.current_step}: balance={self.balance}, shares_held={self.shares_held}, price={price}, reward={reward}")
        # else: Hold
        self.current_step += 1
        self.episode_steps += 1
        if self.current_step >= len(self.df) or self.episode_steps >= self.max_steps_per_episode:
            done = True
        if done:
            # Liquidate any remaining position
            if self.shares_held > 0:
                self.balance += price
                reward += price - self.prev_buy_price if self.prev_buy_price is not None else 0
                self.shares_held = 0
                self.prev_buy_price = None
                print(f"[DONE] Liquidate at step {self.current_step}: balance={self.balance}, shares_held={self.shares_held}, price={price}, reward={reward}")
            self.total_profit = self.balance - self.initial_balance
            self._write_actions_log()
        print(f"[STEP] step={self.current_step}, action={action}, balance={self.balance}, shares_held={self.shares_held}, price={price}, reward={reward}, done={done}")
        return self._get_observation(), reward, done, False, {}

    def _write_actions_log(self):
        with open(self.log_actions_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'action', 'balance', 'shares_held', 'price'])
            writer.writerows(self.actions)

    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Shares held: {self.shares_held}, Total profit: {self.total_profit}') 