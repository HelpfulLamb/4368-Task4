import random
from collections import defaultdict, deque, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class PDWorld:
  def __init__(self):
    self.grid_size = (5, 5)
    self.dropoff_locations = [(1, 1), (1, 5), (3, 3), (5, 5)]
    self.pickup_locations = [(3, 5), (4, 2)]
    self.initial_F = (1, 3)
    self.initial_M = (5, 3)
    self.total_blocks = 10
    self.blocks_at_pickup = {p:5 for p in self.pickup_locations}  # 5 blocks at each pickup location
    self.blocks_at_dropoff = {p:0 for p in self.dropoff_locations}  # each dropoff location starts with 0 blocks
    self.operators = [0, 1, 2, 3, 4, 5]  # 0 = North, 1 = South, 2 = East, 3 = West, 4 = Pickup, 5 = Dropoff
    self.F_pos = self.initial_F
    self.M_pos = self.initial_M
    self.F_carry = False
    self.M_carry = False
  
  def reset(self):  # resets a full episode
    self.reset_world()
    self.F_pos = self.initial_F  # puts F agent back to original spot
    self.M_pos = self.initial_M  # puts M agent back to original spot
    self.F_carry = False  # clear the carry
    self.M_carry = False  # clear the carry
    return self.encode_state()

  def reset_world(self):  # resets blocks in the world
    self.blocks_at_pickup = {p:5 for p in self.pickup_locations}
    self.blocks_at_dropoff = {p:0 for p in self.dropoff_locations}
  
  def terminal(self):  # check if all blocks have been delivered
    total_delivered = sum(self.blocks_at_dropoff.values())
    return total_delivered >= self.total_blocks
  
  def valid_position(self, pos):  # check if the movement position is within the grid
    return 1 <= pos[0] <= self.grid_size[0] and 1 <= pos[1] <= self.grid_size[1]
  
  def aplop(self, agent_pos, carrying_block, other_agent_pos):
    applicable = []
    moves = {
      0: (agent_pos[0] - 1, agent_pos[1]),  # agent moves North
      1: (agent_pos[0] + 1, agent_pos[1]),  # agent moves South
      2: (agent_pos[0], agent_pos[1] + 1),  # agent moves East
      3: (agent_pos[0], agent_pos[1] - 1)  # agent moves West
    }

    for op, new_pos in moves.items():
      if self.valid_position(new_pos) and new_pos != other_agent_pos:
        applicable.append(op)
    
    # Pickup operator, if the agent is in a pickup cell that is not empty
    if not carrying_block and agent_pos in self.pickup_locations and self.blocks_at_pickup[agent_pos] > 0:
      applicable.append(4)
    
    # Dropoff operator, if the agent is in a dropoff cell that is not full
    if carrying_block and agent_pos in self.dropoff_locations and self.blocks_at_dropoff[agent_pos] < 5:
      applicable.append(5)
    return applicable
  
  def apply(self, operator, agent_pos, carrying_block, other_agent_pos):
    reward = 0  # no reward if there is no state change
    new_pos = agent_pos
    new_carrying = carrying_block

    if operator in (0, 1, 2, 3):
      delta = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
      pos_change = (agent_pos[0] + delta[operator][0], agent_pos[1] + delta[operator][1])
      if self.valid_position(pos_change) and (other_agent_pos is None or pos_change != other_agent_pos):
        new_pos = pos_change
        reward = -1  # reward/cost for moving is -1
    elif operator == 4:  # Pickup
      if(not carrying_block) and agent_pos in self.pickup_locations and self.blocks_at_pickup[agent_pos] > 0:
        self.blocks_at_pickup[agent_pos] -= 1  # decrement the number of boxes from the pickup location
        new_carrying = True  # agent is now carrying block
        reward = +13  # reward for picking up is +13
    elif operator == 5:  # Dropoff
      if carrying_block and agent_pos in self.dropoff_locations and self.blocks_at_dropoff[agent_pos] < 5:
        self.blocks_at_dropoff[agent_pos] += 1  # increment the number of boxes in the dropoff location
        new_carrying = False  # agent is no longer carrying block
        reward = +13  # reward for dropoff is +13
    return new_pos, new_carrying, reward
  
  def encode_state(self):
    Fx, Fy = self.F_pos
    Mx, My = self.M_pos
    Fcarry = int(self.F_carry)
    Mcarry = int(self.M_carry)
    # pick_up = tuple(self.blocks_at_pickup[p] for p in self.pickup_locations)
    # drop_off = tuple(self.blocks_at_dropoff[p] for p in self.dropoff_locations)
    return (Fx, Fy, Fcarry, Mx, My, Mcarry) # + pick_up + drop_off
  
  def get_agent_state(self, agent):
    if agent == 'F':
      return (self.F_pos[0], self.F_pos[1], self.M_pos[0], self.M_pos[1], int(self.F_carry))
    else:
      return (self.M_pos[0], self.M_pos[1], self.F_pos[0], self.F_pos[1], int(self.M_carry))
  
  def step(self, action, agent):  # applies one agents action and updates world
    # pick the agent that is taking action
    if agent == 'F':
      pos, carry, other = self.F_pos, self.F_carry, self.M_pos
    else:
      pos, carry, other = self.M_pos, self.M_carry, self.F_pos
    new_pos, new_carry, reward = self.apply(action, pos, carry, other)  # execute action

    # update agent state in the world
    if agent == 'F':
      self.F_pos, self.F_carry = new_pos, new_carry
    else:
      self.M_pos, self.M_carry = new_pos, new_carry
    
    # checks to see if the current episode is done
    done = self.terminal()
    return self.get_agent_state(agent), reward, done, {}  # returns the next observation

class TwoAgentRLSystem:
  def __init__(self):
    self.world = PDWorld()
    self.alpha = 0.3  # the default learning rate for experiment 1
    self.gamma = 0.5  # the default discount factor for experiment 1
    self.q_tables = {
      'F': defaultdict(lambda: np.zeros(6)),
      'M': defaultdict(lambda: np.zeros(6))
    }
    self.performance_history = []
    self.update_log = []  # for logging during execution, this helps with debugging

  def preferred_action(self, carrying, applicable_op):
    # prefer dropoff if carrying, otherwise prefer pickup if not carrying
    if carrying and 5 in applicable_op:
      return 5
    if(not carrying) and 4 in applicable_op:
      return 4
    return None
  
  def PRandom(self, agent, state, applicable_op):
    if not applicable_op:
      return None
    carrying = bool(state[4])
    pref_action = self.preferred_action(carrying, applicable_op)
    if pref_action is not None:
      return pref_action
    return random.choice(applicable_op)  # otherwise choose an operator randomly

  def PExploit(self, agent, state, applicable_op):
    if not applicable_op:
      return None
    carrying = bool(state[4])
    pref_action = self.preferred_action(carrying, applicable_op)
    # prefer goal advancing action if available
    if pref_action is not None:
      return pref_action
    # otherwise get Q-value
    q_values = self.q_tables[agent][state]
    best_value = max(q_values[a] for a in applicable_op)  # get the highest q-value to apply applicable operator
    best_ops = [a for a in applicable_op if q_values[a] == best_value]
    if random.random() < 0.8:
      return random.choice(best_ops)  # choose among the best, this is the tie break
    else:
      other = [a for a in applicable_op if a not in best_ops]
      # choose a different applicable op if possible
      return random.choice(other) if other else random.choice(applicable_op)

  def PGreedy(self, agent, state, applicable_op):
    if not applicable_op:
      return None
    carrying = bool(state[4])
    pref_action = self.preferred_action(carrying, applicable_op)
    if pref_action is not None:
      return pref_action
    q_values = self.q_tables[agent][state]
    best_value = max(q_values[a] for a in applicable_op)  # get the highest q-value to apply applicable operator
    best_ops = [a for a in applicable_op if q_values[a] == best_value]
    return random.choice(best_ops)  # tie break
  
  def get_applicable_actions(self, agent):  # helper function to access applicable actions
    if agent == 'F':
      return self.world.aplop(self.world.F_pos, self.world.F_carry, self.world.M_pos)
    else:
      return self.world.aplop(self.world.M_pos, self.world.M_carry, self.world.F_pos)

  def select_actions(self, behavior, agent, state, applicable_op):
    if behavior == "PRandom":
      return self.PRandom(agent, state, applicable_op)
    elif behavior == "PExploit":
      return self.PExploit(agent, state, applicable_op)
    elif behavior == "PGreedy":
      return self.PGreedy(agent, state, applicable_op)
  
  # training Q-learn
  def QLearn(self, agent, state, action, reward, next_state, next_applicable_op):
    current_q = self.q_tables[agent][state][action]
    if next_applicable_op:
      max_next_q = max(self.q_tables[agent][next_state][a] for a in next_applicable_op)
    else:
      max_next_q = 0
    q_before = current_q
    # Q-Learning Formula
    new_q = (1 - self.alpha) * current_q + self.alpha*(reward + self.gamma * max_next_q)
    # update Q-table
    self.q_tables[agent][state][action] = new_q
    # update logs for visualizations
    self.update_log.append({
      "agent": agent,
      "s": state,
      "a": action,
      "r": reward,
      "s_prime": next_state,
      "max_next_q": max_next_q,  # Q-Learning uses max_a' Q(s', a')
      "q_before": q_before,
      "q_after": new_q
    })

  # training SARSA
  def Sarsa(self, agent, state, action, reward, next_state, next_action):
    current_q = self.q_tables[agent][state][action]
    next_q = self.q_tables[agent][next_state][next_action] if next_action is not None else 0
    q_before = current_q
    # SARSA Formula
    new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
    # update Q-table
    self.q_tables[agent][state][action] = new_q
    # update the logs for visualizations
    self.update_log.append({
      "agent": agent,
      "s": state,
      "a": action,
      "r": reward,
      "s_prime": next_state,
      "next_q": next_q,  # SARSA uses Q(a', s')
      "q_before": q_before,
      "q_after": new_q
    })
  
  # Q-Learning off-policy
  def train_qlearn(self, phases, n_steps, max_steps_per_episode, seed, change_pickups_at_terminal):
    if seed is not None:
      random.seed(seed)
      np.random.seed(seed)

    # helper to decide which agent is acting
    def acting_agent(t):
      return 'F' if (t % 2 == 0) else 'M'
    
    world = self.world
    world.reset()

    total_steps = 0
    episode_steps = 0
    episode_count = 0
    terminal_hits = 0
    changed_layout = False
    episode_rewards = 0
    episode_manhattan_dist = []

    phase_count = 0
    steps_in_phase = 0
    behavior = phases[phase_count]["behavior"]  # PRandom, PExploit, or PGreedy

    pending = {'F': None, 'M': None}  # per-agent pending transition: (s_prev, a_prev, reward_prev), or None
    
    while total_steps < n_steps:
      # roll into the next phase
      if steps_in_phase >= phases[phase_count]["steps"]:
        phase_count += 1
        if phase_count >= len(phases):
          break
        steps_in_phase = 0
        behavior = phases[phase_count]["behavior"]
      agent = acting_agent(total_steps)

      # observe the decision state for the current agent
      s = world.get_agent_state(agent)
      A = self.get_applicable_actions(agent)
      a = self.select_actions(behavior, agent, s, A)

      # complete previous transition for the current agent
      if pending[agent] is not None:
        s_prev, a_prev, reward_prev = pending[agent]
        self.QLearn(agent, s_prev, a_prev, reward_prev, s, A)
        pending[agent] = None

      # execute current agents action
      s_next, reward, done, _ = world.step(a, agent)
      episode_rewards += reward  # accumulate rewards
      episode_manhattan_dist.append(self.manhattan_distance())  # record manhattan distance for coordination analysis

      # start new pending transition
      pending[agent] = (s, a, reward)

      # bookkeeping
      total_steps += 1
      steps_in_phase += 1
      episode_steps += 1

      # episode boundary
      if done or episode_steps >= max_steps_per_episode:
        episode_count += 1
        if done:
          terminal_hits += 1

        avg_manhattan = sum(episode_manhattan_dist) / len(episode_manhattan_dist) if episode_manhattan_dist else 0
        self.performance_history.append({
          "episode": episode_count,
          "steps": episode_steps,
          "total_rewards": episode_rewards,
          "avg_manhattan": avg_manhattan,
          "phase_index": phase_count,
          "behavior": behavior,
          "blocks_delivered": sum(world.blocks_at_dropoff.values()),
          "terminal_hits": terminal_hits,
          "steps_so_far": total_steps
        })
        
        # finalize all pending transitions
        for current_agent in ('F', 'M'):
          if pending[current_agent] is not None:
            s_prev, a_prev, reward_prev = pending[current_agent]
            self.QLearn(current_agent, s_prev, a_prev, reward_prev, None, None)
            pending[current_agent] = None

        # this is for experiment 4: move pickup locations after k terminals
        if change_pickups_at_terminal is not None:
          if(not changed_layout) and (terminal_hits >= change_pickups_at_terminal["k"]):
            world.pickup_locations[0] = change_pickups_at_terminal["coords"][0]
            world.pickup_locations[1] = change_pickups_at_terminal["coords"][1]
            world.reset_world()  # rebuild counts at new positions
            changed_layout = True
          if terminal_hits >= 6:  # stops after the 6th terminal hit
            print("Reached 6th terminal hit, experiment 4 stops")
            break

        world.reset()
        episode_steps = 0
        episode_rewards = 0
        episode_manhattan_dist = []
  
  def train_sarsa(self, phases, n_steps, max_steps_per_episode, seed, change_pickups_at_terminal):
    if seed is not None:
      random.seed(seed)
      np.random.seed(seed)

    def acting_agent(t):
      return 'F' if (t % 2 == 0) else 'M'
    
    world = self.world
    world.reset()

    total_steps = 0
    episode_steps = 0
    episode_count = 0
    terminal_hits = 0
    changed_layout = False
    episode_rewards = 0
    episode_manhattan_dist = []

    phase_count = 0
    steps_in_phase = 0
    behavior = phases[phase_count]["behavior"]  # PRandom, PExploit, or PGreedy

    pending = {'F': None, 'M': None}  # per-agent pending transition: (s_prev, a_prev, reward_prev), or None

    while total_steps < n_steps:
      if steps_in_phase >= phases[phase_count]["steps"]:
        phase_count += 1
        if phase_count >= len(phases):
          break
        steps_in_phase = 0
        behavior = phases[phase_count]["behavior"]
      agent = acting_agent(total_steps)

      # observe the current decision state for the current agent
      s = world.get_agent_state(agent)
      A = self.get_applicable_actions(agent)
      a = self.select_actions(behavior, agent, s, A)

      # if the current agent has a pending (s_prev, a_prev, reward_prev) finish its SARSA update
      if pending[agent] is not None:
        s_prev, a_prev, reward_prev = pending[agent]
        # on policy next action is the one we just chose (a at state s)
        self.Sarsa(agent, s_prev, a_prev, reward_prev, s, a)
        pending[agent] = None
      
      # execute the current agents action
      s_next, reward, done, _ = world.step(a, agent)
      episode_rewards += reward
      episode_manhattan_dist.append(self.manhattan_distance())

      # start a new pending transition for this agent using the reward we just got
      pending[agent] = (s, a, reward)

      # bookkepping
      total_steps += 1
      steps_in_phase += 1
      episode_steps += 1

      # episode boundary / terminal handling
      if done or episode_steps >= max_steps_per_episode:
        episode_count += 1
        if done:
          terminal_hits += 1

        avg_manhattan = sum(episode_manhattan_dist) / len(episode_manhattan_dist) if episode_manhattan_dist else 0
        self.performance_history.append({
          "episode": episode_count,
          "steps": episode_steps,
          "total_rewards": episode_rewards,
          "avg_manhattan": avg_manhattan,
          "phase_index": phase_count,
          "behavior": behavior,
          "blocks_delivered": sum(world.blocks_at_dropoff.values()),
          "terminal_hits": terminal_hits,
          "steps_so_far": total_steps
        })

        # finalize any pending transitions for both agents
        for current_agent in ('F', 'M'):
          if pending[current_agent] is not None:
            s_prev, a_prev, reward_prev = pending[current_agent]
            self.Sarsa(current_agent, s_prev, a_prev, reward_prev, None, None)
            pending[current_agent] = None

        # this if for experiment 4: move pickups after k terminals
        if change_pickups_at_terminal is not None:
          if(not changed_layout) and (terminal_hits >= change_pickups_at_terminal["k"]):
            world.pickup_locations[0] = change_pickups_at_terminal["coords"][0]
            world.pickup_locations[1] = change_pickups_at_terminal["coords"][1]
            world.reset_world()
            changed_layout = True
          if terminal_hits >= 6:  # stop the experiment after 6 terminals
            print("Reached 6th terminal hit, experiment 4 stops")
            break

        world.reset()
        episode_steps = 0
        episode_rewards = 0
        episode_manhattan_dist = []
  
  def manhattan_distance(self):
    return abs(self.world.F_pos[0] - self.world.M_pos[0]) + abs(self.world.F_pos[1] - self.world.M_pos[1])

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

  def print_update_log(self, agent):  # TODO: use this for the path visualizer 
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

# function to run the entire program
def run_RL():
  rl_runner = ExperimentRunner()
  all_results = []

  # running experiment 1
  print("Experiment 1: Q-learning with different policies")
  
  # two runs, each with different seeds
  for run in [1, 2]:
    print(f"--- run {run} ---")
    rl_runner.system = TwoAgentRLSystem()  # reset system for a fresh start

    # experiment 1.a
    rl_runner.system = TwoAgentRLSystem()  # reset system for a fresh start
    result_1a = rl_runner.experiment_1('a', seed=10 + run, alpha=0.3, gamma=0.5)  # seed updates for each run
    all_results.append(result_1a)
    # plot attractive paths for both agents in different states
    #for agent in ['F', 'M']:
    #  for carrying in [False, True]:
    #    rl_runner.plot_attractive_path(agent, carrying, '1a', run=run, threshold=0)
    rl_runner.plot_learning_curve('1a')

    rl_runner.analyze_learned_paths(result_1a)
    print("\nQ-Table for Experiment 1.a:")
    #rl_runner.print_update_log('F')
    #rl_runner.print_update_log('M')
    rl_runner.print_q_table('F')

    # experiment 1.b
    rl_runner.system = TwoAgentRLSystem()  # reset system for a fresh start
    result_1b = rl_runner.experiment_1('b', seed=10 + run, alpha=0.3, gamma=0.5)
    all_results.append(result_1b)
    rl_runner.analyze_learned_paths(result_1b)
    print("\nQ-Table for Experiment 1.b:")
    #rl_runner.print_update_log('F')
    #rl_runner.print_update_log('M')

    # experiment 1.c
    rl_runner.system = TwoAgentRLSystem()  # reset system for a fresh start
    result_1c = rl_runner.experiment_1('c', seed=10 + run, alpha=0.3, gamma=0.5)
    all_results.append(result_1c)

    # analyze learned path for 1.c
    rl_runner.analyze_learned_paths(result_1c)
    # print Q-table for 1.c for agent F
    print("\nQ-Table for Experiment 1.c:")
    #rl_runner.print_update_log('F')  # replace the current lines with rl_runner.print_q_table('F')
    #rl_runner.print_update_log('M')  # replace the current lines with rl_runner.print_q_table('M')

  # running experiment 2
  print("\nExeriment 2: SARSA learning")
  for run in [1, 2]:
    print(f"--- run {run} ---")
    rl_runner.system = TwoAgentRLSystem()

    # experiment 2
    result_2 = rl_runner.experiment_2(seed=10 + run, alpha=0.3, gamma=0.5)
    all_results.append(result_2)

    # TODO: analyze the learned path for experiment 2
    #rl_runner.analyze_learned_paths(result_2)

    # print (one of the) Q-table for exp.2
    print("\nQ-Table for Experiment 2:")
    
    # TODO: print Q-Table for either agent F or M

  # running experiment 3
  print("\nExperiment 3: Tuning learning rate")
  learning_rates = [0.15, 0.45]
  for lr in learning_rates:
    for run in [1, 2]:  # two runs for each learning rate
      rl_runner.system = TwoAgentRLSystem()

      # experiment 3
      result_3 = rl_runner.experiment_3(seed=10 + run, alpha=lr, gamma=0.5)
      all_results.append(result_3)

      # TODO: analyze learned path for each alpha
      #rl_runner.analyze_learned_paths(result_3)

      # TODO: print Q-table for both F and M agent
      print("\nQ-Table for Experiment 3:")

  # running experiment 4
  print("\nExperiment 4: Adaptation to changes in pickup locations")
  for run in [1, 2]:
    rl_runner.system = TwoAgentRLSystem()

    # experiment 4
    result_4 = rl_runner.experiment_4(seed=10 + run, alpha=0.3, gamma=0.5)
    all_results.append(result_4)

    # TODO: analyze learned path
    #rl_runner.analyze_learned_paths(result_4)

    # TODO: print Q-table for experiment 4
    print("\nQ-Table for Experiment 4:")
  
  # TODO: possible final analysis comparison

  return all_results

# run the program
if __name__ == "__main__":
  results = run_RL()
