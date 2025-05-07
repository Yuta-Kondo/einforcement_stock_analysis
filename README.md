# Reinforcement Learning for S&P 500 and Forex (EUR/USD) Trading

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This project demonstrates the application of reinforcement learning (RL) to financial trading. RL agents are trained to trade the S&P 500 index and the EUR/USD forex pair using historical price data, highlighting both the promise and challenges of RL in real-world markets.

## Features
- Custom OpenAI Gymnasium trading environment
- DQN agent (Stable Baselines3)
- Experiments on both S&P 500 and Forex (EUR/USD)
- Automated data collection, training, evaluation, and plotting
- Professional report with results and analysis

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yuta-Kondo/einforcement_stock_analysis.git
   cd einforcement_stock_analysis
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download data:**
   ```bash
   python src/data_collection.py
   ```

## Usage
- **Train the agent:**
  ```bash
  python src/train_agent.py
  ```
- **Plot price movements:**
  ```bash
  python src/plot_sp500.py
  python src/plot_forex.py
  ```
- **View results:**
  See generated PNG files and the [report.md](report.md) for detailed results and analysis.

## Results
- RL agents can learn to "buy and hold" in trending markets (S&P 500), but struggle to learn profitable strategies in mean-reverting or volatile markets (Forex) with simple state/reward structures.
- The project includes six key figures: price movement, training reward, and final profits per episode for both S&P 500 and Forex.

## License
MIT 