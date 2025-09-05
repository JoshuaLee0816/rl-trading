import matplotlib.pyplot as plt

def plot_entropy(entropy_log):
    plt.figure(figsize=(8,5))
    plt.plot(entropy_log, label="Policy Entropy")
    plt.xlabel("Update Steps")
    plt.ylabel("Entropy")
    plt.title("PPO Exploration Trend")
    plt.legend()
    plt.grid(True)
    plt.show()
