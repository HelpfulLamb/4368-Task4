import math
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
  def __init__(self):
    pass
  
  # plotting the learning curves of each RL Algo for the experiments
  def plot_learning_curve(self, performance_history, agent, run, title="Learning Curve"):
    episodes = [ep["episode"] for ep in performance_history]
    rewards = [ep["total_rewards"] for ep in performance_history]
    blocks = [ep["blocks_delivered"] for ep in performance_history]
    manhattan = [ep["avg_manhattan"] for ep in performance_history]
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(12,10))
    ax1.plot(episodes, rewards, 'b-', alpha=0.6)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title(f"{title} for Agent {agent}: Run {run}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, blocks, 'g-', alpha=0.6)
    ax2.set_ylabel('Blocks Delivered')
    ax2.grid(True, alpha=0.3)

    ax3.plot(episodes, manhattan, 'r-', alpha=0.6)
    ax3.set_ylabel('Avg Manhattan Distance')
    ax3.set_xlabel('Episode')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
  
  # Q-Table Arrows
  def plot_q_arrows(self, q_table, agent="F", grid_size=(5,5), title="Q-Table Arrows"):
    moves = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}  # N, S, E, W
    width, height, = grid_size
    fig, ax = plt.subplots()
    ax.set_xlim(0.5, width + 0.5)
    ax.set_ylim(0.5, height + 0.5)
    ax.set_xticks(range(1, width + 1))
    ax.set_yticks(range(1, height + 1))
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()

    # aggregate best Q per cell
    per_cell = {}
    for state, q in q_table.items():
      if len(state) < 5:
        continue
      pos = (state[0], state[1]) if agent == 'F' else (state[2], state[3])
      move_qs = q[:4]
      best_action = int(np.argmax(move_qs))
      best_value = float(move_qs[best_action])
      if pos not in per_cell or best_value > per_cell[pos][1]:
        per_cell[pos] = (best_action, best_value)
    
    # draw arrows
    for (r,c), (a, qv) in per_cell.items():
      dx, dy = moves[a]
      ax.arrow(c, r, dx*0.3, head_width=0.15, length_includes_head=True)
      ax.text(c, r, f"{qv:.1f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title(f"{title} (agent = {agent})")
    plt.show()
  
  def plot_steps_per_episode(self, performance_history, agent, run, title="Steps per Episode"):
    episodes = [ep['episode'] for ep in performance_history]
    steps = [ep['steps'] for ep in performance_history]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(episodes, steps, color="purple", alpha=0.7, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)
    # optional smoothing
    plt.show()

