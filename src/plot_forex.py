import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('forex.csv', skiprows=2)
    df = df.rename(columns={'Unnamed: 1': 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.title('EUR/USD Close Price')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.savefig('price_forex.png')
    plt.close()

if __name__ == '__main__':
    main() 