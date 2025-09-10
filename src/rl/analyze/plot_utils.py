import numpy as np
import matplotlib.pyplot as plt

def plot_reward_curve(all_rewards, outdir):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, len(all_rewards) + 1)
    ax.plot(x, all_rewards, label="Annualized Return (%)", color="#1f77b4", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Baseline = 0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Annualized Return (%)")
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("Training Progress (PPO vs Baseline=0)")
    fig.tight_layout()
    fig.savefig(outdir / "reward_curve.png")
    plt.close(fig)


def plot_entropy_curve(entropy_log, outdir):
    if len(entropy_log) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, len(entropy_log) + 1)
    ax.plot(x, entropy_log, label="Entropy", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("Policy Entropy")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_curve.png")
    plt.close(fig)


def plot_loss_curve(actor_loss_log, critic_loss_log, outdir, window=100):
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode="valid")

    actor_smooth = moving_average(actor_loss_log, window=window)
    critic_smooth = moving_average(critic_loss_log, window=window)
    x_smooth = range(1, len(actor_smooth) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 左邊 Y 軸 (Actor Loss)
    ax1.plot(x_smooth, actor_smooth, label="Actor Loss (smoothed)", color="blue", linewidth=1)
    ax1.set_xlabel("Update Step")
    ax1.set_ylabel("Actor Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # 右邊 Y 軸 (Critic Loss)
    ax2 = ax1.twinx()
    ax2.plot(x_smooth, critic_smooth, label="Critic Loss (smoothed)", color="red", linewidth=1)
    ax2.set_ylabel("Critic Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.suptitle("Actor & Critic Loss Curve (Smoothed)")
    fig.tight_layout()
    fig.savefig(outdir / "actor_critic_loss_curve.png")
    plt.close(fig)
