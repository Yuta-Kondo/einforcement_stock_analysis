import pandas as pd
import matplotlib.pyplot as plt

# Read the actions log
log = pd.read_csv('actions_log.csv')

# Count each action
action_counts = log['action'].value_counts().sort_index()
print('Action counts:')
print(f"Hold (0): {action_counts.get(0, 0)}")
print(f"Buy  (1): {action_counts.get(1, 0)}")
print(f"Sell (2): {action_counts.get(2, 0)}")

# Plot actions over time
plt.figure(figsize=(12, 4))
plt.plot(log['step'], log['action'], marker='o', linestyle='-', markersize=2)
plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Agent Actions Over Time')
plt.grid(True)
plt.show() 