import numpy as np
import random
import matplotlib.pyplot as plt

class ThresholdRLAgent:
    def __init__(self):
        # Discrete threshold space
        self.thresholds = np.arange(0.3, 0.95, 0.05)

        # Q-table (state simplified → threshold only)
        self.q_table = {t: 0.0 for t in self.thresholds}

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

        self.rewards_history = []
        self.threshold_history = []

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.choice(self.thresholds)
        return max(self.q_table, key=self.q_table.get)

    def update(self, threshold, reward):
        old_value = self.q_table[threshold]
        best_next = max(self.q_table.values())

        new_value = old_value + self.alpha * (
            reward + self.gamma * best_next - old_value
        )

        self.q_table[threshold] = new_value

        self.rewards_history.append(reward)
        self.threshold_history.append(threshold)

    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * 0.995)

    def plot_learning(self, save_path="experiments/results/rl_learning.png"):
        plt.figure()
        window = 10
        smoothed = np.convolve(self.rewards_history,
                            np.ones(window)/window,
                            mode='valid')
        plt.plot(smoothed)
        plt.title("RL Learning Curve (Reward vs Steps)")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.savefig(save_path)
        plt.close()

    def get_best_threshold(self):
        return max(self.q_table, key=self.q_table.get)