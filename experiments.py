import matplotlib.pyplot as plt
from collections import deque, OrderedDict, defaultdict
import numpy as np
from two_agent_sys import TwoAgentRLSystem

class ExperimentRunner:
  def __init__(self):
    self.system = TwoAgentRLSystem()
  
  def get_performance_summary(self, experiment_name):
    if not self.system.performance_history:
      return {"error": "No performance data recorded", "experiment_name": experiment_name}
    
    total_episodes = len(self.system.performance_history)
    total_reward = sum(ep["total_rewards"] for ep in self.system.performance_history)
    avg_reward = total_reward / total_episodes
    avg_steps = sum(ep["steps"] for ep in self.system.performance_history) / total_episodes
    avg_manhattan_dist = sum(ep["avg_manhattan"] for ep in self.system.performance_history) / total_episodes
    terminals_reached = self.system.performance_history[-1]["terminal_hits"]
    last_phase = self.system.performance_history[-1]["phase_index"]
    last_behavior = self.system.performance_history[-1]["behavior"]

    return {
      "experiment_name": experiment_name,
      "total_episodes": total_episodes,
      "total_rewards": total_reward,
      "avg_rewards_per_ep": avg_reward,
      "avg_steps_per_ep": avg_steps,
      "average_manhattan_distance": avg_manhattan_dist,
      "terminals_reached": terminals_reached,
      "last_phase_index": last_phase,
      "last_behavior": last_behavior
    }
  
  def experiment_1(self, variant, seed, alpha, gamma):
    self.system.performance_history.clear()  # we clear here so the summary reflects only the current experiment
    self.system.update_log.clear()  # clear so the logs only show current experiment
    print(f"Running Experiment 1.{variant}")

    # setting alpha and gamma is not necessary because for parts a,b,c we dont alter the parameters
    self.system.alpha = alpha  # not totally necessary since the default is 0.3
    self.system.gamma = gamma  # not totally necessary since the default is 0.5

    if variant == 'a':  # total steps is 8000
      phases = [
        {"name": "random_500", "behavior": "PRandom", "steps": 500},
        {"name": "random_7500", "behavior": "PRandom", "steps": 7500}
      ]
    elif variant == 'b':  # total steps is 8000
      phases = [
        {"name": "random_500", "behavior": "PRandom", "steps": 500},
        {"name": "greedy_7500", "behavior": "PGreedy", "steps": 7500}
      ]
    else:  # variant = c, total steps is 8000
      phases = [
        {"name": "random_500", "behavior": "PRandom", "steps": 500},
        {"name": "exploit_7500", "behavior": "PExploit", "steps": 7500}
      ]
    
    # running Q-learning first
    self.system.train_qlearn(phases=phases, n_steps=8000, max_steps_per_episode=1000, seed=seed, change_pickups_at_terminal=None)
    return self.get_performance_summary(f"1{variant}")

  def experiment_2(self, seed, alpha, gamma):
    self.system.performance_history.clear()
    self.system.update_log.clear()

    print("Running Experiment 2")

    # this is not totally necessary since experiment 2 repeats 1.c but doesnt change the parameters, so we can use the defaults
    self.system.alpha = alpha  # not totally necessary since the parameter is the same as experiement 1
    self.system.gamma = gamma  # not totally necessary since the parameter is the same as experiement 1
    phases = [
      {"name": "random_500", "behavior": "PRandom", "steps": 500},
      {"name": "exploit_7500", "behavior": "PExploit", "steps": 7500}
    ]
    self.system.train_sarsa(phases=phases, n_steps=8000, max_steps_per_episode=1000, seed=seed, change_pickups_at_terminal=None)
    return self.get_performance_summary("2")

  def experiment_3(self, seed, alpha, gamma):
    self.system.performance_history.clear()
    self.system.update_log.clear()
    print(f"Running Experiment 3: alpha = {alpha}")

    # setting a new alpha parameter to test
    self.system.alpha = alpha  # changing the alpha parameter only
    self.system.gamma = gamma  # not totally necessary since this parameter is not changed
    
    # re-running experiment 1.c
    phases = [
      {"name": "random_500", "behavior": "PRandom", "steps": 500},
      {"name": "exploit_7500", "behavior": "PExploit", "steps": 7500}
    ]

    self.system.train_qlearn(phases=phases, n_steps=8000, max_steps_per_episode=1000, seed=seed, change_pickups_at_terminal=None)
    return self.get_performance_summary("3")

  def experiment_4(self, seed, alpha, gamma):
    self.system.performance_history.clear()
    self.system.update_log.clear()
    print("Running experiment 4")

    # not totally necessary since the alpha and gamma will just be the defaults
    self.system.alpha = alpha
    self.system.gamma = gamma

    phases = [
      {"name": "random_500", "behavior": "PRandom", "steps": 500},
      {"name": "exploit_7500", "behavior": "PExploit", "steps": 7500}
    ]

    # run SARSA but when we hit 3rd terminal state, we change the pickup locations only
    self.system.train_sarsa(phases=phases, n_steps=8000, max_steps_per_episode=1000, seed=seed, change_pickups_at_terminal={"k": 3, "coords": [(1,2), (4,5)]})
    return self.get_performance_summary("4")
  
  # TODO: may add some more details idk
  def assess_coordination(self, avg_manhattan):
    if avg_manhattan >= 5.0:  # agents stay far apart
      return "Excellent Coordination - agents work in different areas"  # maybe something else could be returned
    elif avg_manhattan >= 3.5:  # moderate distance
      return "Good Coordination - some separation between agents"  # maybe something else could be returned
    else:  # close proximity
      return "Poor coordination - agents frequently block each other"  # maybe something else could be returned
  
  def bfs_dist(self, src, dst):
    # 4 neighbor BFS on a 5x5 grid
    sx, sy = src
    tx, ty = dst
    q = deque([(sx, sy, 0)])
    seen = {(sx, sy)}
    inside = lambda x, y: 1 <= x <= 5 and 1 <= y <= 5
    while q:
      x, y, d = q.popleft()
      if(x,y) == (tx,ty):
        return d
      for dx, dy in [(-1,0),(1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if inside(nx, ny) and (nx, ny) not in seen:
          seen.add((nx, ny))
          q.append((nx, ny, d+1))
    return None  # means it was unreachable, though it should not happen on an empty grid
  
  def greedy_path_len(self, agent, start_pos, carrying, ignore_blocking=True):
    world = self.system.world
    q_table = self.system.q_tables[agent]

    # simulate only the acting agents motion, ignoring the other agent
    pos = start_pos
    steps = 0
    max_steps = 100
    while steps < max_steps:
      # build an agent tuple
      if agent == 'F':
        s = (pos[0], pos[1], world.M_pos[0], world.M_pos[1], int(carrying))
        other_pos = None if ignore_blocking else world.M_pos
      else:
        s = (pos[0], pos[1], world.F_pos[0], world.F_pos[1], int(carrying))
        other_pos = None if ignore_blocking else world.F_pos
      
      # get applicable movement actions (ignoring PU and DO)
      applicable = world.aplop(pos, carrying, other_pos)
      move_ops = [a for a in applicable if a in (0,1,2,3)]
      if not move_ops:
        break  # no valid moves, stop early
      
      # choose best move among the movement operators
      q = q_table[s]
      best_move = max(move_ops, key = lambda a: q[a])

      # apply the move safely
      delta = {0:(-1,0), 1:(1,0), 2:(0,1), 3:(0,-1)}
      new_pos = (pos[0]+delta[best_move][0], pos[1]+delta[best_move][1])
      if not world.valid_position(new_pos):
        break  # although we already have a safety net to catch invalids, this ensures it even more
      pos = new_pos
      steps += 1

      # stop when we hit a relevant goal
      if carrying and pos in world.dropoff_locations:
        break
      if not carrying and pos in world.pickup_locations:
        break
    return steps if steps < max_steps else None

  def analyze_learned_paths(self, label, ingnore_blocking=True):
    rows = []
    stuck = []

    def probe(agent, src, carrying):
      L = self.greedy_path_len(agent, src, carrying, ignore_blocking=ingnore_blocking)
      return L
    
    # Carry: from each pickup to nearest dropoff
    for agent in ['F', 'M']:
      for src in self.system.world.pickup_locations:
        best_baseline = min(self.bfs_dist(src, d) for d in self.system.world.dropoff_locations)
        L = probe(agent, src, carrying=True)
        if L is None:
          stuck.append((agent, "carry", src))
        else:
          rows.append((label, agent, "carry", src, L, best_baseline, L - best_baseline))

      # Empty: from each dropoff to nearest pickup
      for src in self.system.world.dropoff_locations:
        best_baseline = min(self.bfs_dist(src, p) for p in self.system.world.pickup_locations)
        L = probe(agent, src, carrying=False)
        if L is None:
          stuck.append((agent, "empty", src))
        else:
          rows.append((label, agent, "empty", src, L, best_baseline, L - best_baseline))
    
    rows.sort(key=lambda r: r[6], reverse=True)  # sort by gap desc
    print("agent, mode, src, learned, shortest, gap, optimal%")
    opt_acc = []
    for _, ag, mode, src, L, B, G in rows:
      optimal_pct = (B / L * 100.0) if L > 0 else 0.0
      opt_acc.append(optimal_pct)
      print(f"{ag}, {mode}, {src}, {L}, {B}, {G:+}, {optimal_pct:5.1f}")
    if rows:
      avg_gap = sum(r[6] for r in rows) / len(rows)
      avg_opt = sum(opt_acc) / len(opt_acc)
      print(f"\nSummary: n = {len(rows)}, avg gap = {avg_gap:+.2f}, avg optimal% = {avg_opt:.1f}")
    if stuck:
      print("\nStuck cases (no valid greedy path within cap):")
      for ag, mode, src in stuck:
        print(f"- {ag}, {mode}, {src}")

  # build a map of attractive moves for a given agent and carrying flag
  def find_attractive_paths(self, agent, carrying, *, min_visits = 3, threshold=None, agg="max"):
    q_table = self.system.q_tables[agent]
    pos_buckets = defaultdict(list)
    for state, q_values, in q_table.items():
      if not (isinstance(state, tuple) and len(state) >= 5):
        continue
      if state[4] != int(carrying):
        continue
      pos = (state[0], state[1])
      q_mov = np.array(q_values[:4], dtype=float)  # first 4 entries are movement actions
      # skip rows that are completely 0 for a cleaner map
      if np.allclose(q_mov, 0.0):
        continue
      pos_buckets[pos].append(q_mov)
    
    # aggregate per position
    pos_stats = {}
    agg_values = []
    for pos, rows in pos_buckets.items():
      row_arr = np.vstack(rows)
      count = row_arr.shape[0]
      if count < min_visits:
        continue  # not enough evidence at the cell
      # aggregate movement Qs across constributing states
      mean_qs = row_arr.mean(axis=0)
      max_qs = row_arr.max(axis=0)

      # choose aggregation for ranking cells
      if agg == "mean":
        cell_vec = mean_qs
      else:
        cell_vec = max_qs
      
      best_action = int(np.argmax(cell_vec))
      best_q = float(cell_vec[best_action])
      pos_stats[pos] = {
        "best_action": best_action,
        "best_q": best_q,
        "count": int(count),
        "mean_q": float(mean_qs[best_action]),
        "agg_q": best_q
      }
      agg_values.append(best_q)
    if not pos_stats:
      return OrderedDict()  # nothing attractive was found
    if threshold is None and agg_values:
      threshold = float(np.percentile(agg_values, 75))
    filtered = [(pos, st) for pos, st in pos_stats.items() if st["agg_q"] >= threshold]
    filtered.sort(key=lambda kv: (kv[1]["agg_q"], kv[1]["count"]), reverse=True)
    return OrderedDict(filtered)

  def print_update_log(self, agent):
    batch_size = 1000
    log = [entry for entry in self.system.update_log if entry["agent"] == agent]
    if not log:
      print(f"No updates found for agent {agent}")
      return
    print(f"\nRecent Q-learning updates for agent {agent}")
    print(f"{'s':<27} {'a':<4} {'s*':<27} {'R':>3} {'maxQ[s*,a*]':>12} {'Q_before':>10} {'Q_after':>10}")
    print("-" * 100)
    for i in range(0, len(log), batch_size):
      chunk = log[i:i+batch_size]
      print(f"\nShowing updates {i+1} to {i+len(chunk)}")
      for entry in chunk:
        s = str(entry["s"])
        a = ["N","S","E","W","PU","DO"][entry["a"]]
        s_prime = str(entry["s_prime"])
        R = entry["r"]
        maxQ = entry["max_next_q"]
        qb = entry["q_before"]
        qa = entry["q_after"]
        print(f"{s:<27} {a:<4} {s_prime:<27} {R:>3} {maxQ:>12.3f} {qb:>10.3f} {qa:>10.3f}")
  
  def print_q_table(self, agent, max_states=100):
    q = self.system.q_tables[agent]
    header = ["N", "S", "E", "W", "PU", "DO"]
    print(f"\nQ-Table for agent {agent} (non-zero states only, capped to {max_states})")
    print("-" * 80)
    shown = 0
    for s, vals in q.items():
      if not any(abs(v) > 1e-6 for v in vals):
        continue  # skip all-zero Qs
      print(f"{s} -> {dict(zip(header, [f'{v:6.2f}' for v in vals]))}")
      shown += 1
      if shown >= max_states:
        print(f"... truncated after {max_states} states ...")
        break

  def plot_attractive_path(self, agent, carrying, experiment_name, run, threshold=None, arrow_stride=1, show_values=True):
    pass

  def plot_learning_curve(self, experiment_name):
    """Plot learning curve showing rewards and blocks delivered over time"""
    if not self.system.performance_history:
        print("No performance data to plot")
        return
    
    episodes = [ep["episode"] for ep in self.system.performance_history]
    rewards = [ep["total_rewards"] for ep in self.system.performance_history]
    blocks = [ep["blocks_delivered"] for ep in self.system.performance_history]
    manhattan = [ep["avg_manhattan"] for ep in self.system.performance_history]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Rewards
    ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
    ax1.set_title(f'{experiment_name} - Learning Curve')
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Blocks delivered
    ax2.plot(episodes, blocks, 'g-', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Blocks Delivered')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Manhattan distance (coordination)
    ax3.plot(episodes, manhattan, 'r-', alpha=0.7, linewidth=1)
    ax3.set_ylabel('Avg Manhattan Distance')
    ax3.set_xlabel('Episode')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()