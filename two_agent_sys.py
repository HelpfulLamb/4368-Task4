import random
from collections import defaultdict
import numpy as np
from pdworld import PDWorld

class TwoAgentRLSystem:
  def __init__(self):
    self.world = PDWorld()
    self.alpha = 0.3  # the default learning rate for experiment 1
    self.gamma = 0.5  # the default discount factor for experiment 1
    self.q_tables = {
      'F': defaultdict(lambda: np.zeros(6)),
      'M': defaultdict(lambda: np.zeros(6))
    }
    self.segment_logs = {'F': [], 'M': []}  # list of paths
    self.seg_active = {'F': None, 'M': None}  # current in progress path while carrying
    self.performance_history = []
    self.update_log = []  # for logging during execution, this helps with debugging

  def preferred_action(self, carrying, applicable_op):  # check for a preferred action
    # prefer dropoff if carrying, otherwise prefer pickup if not carrying
    if carrying and 5 in applicable_op:
      return 5
    if(not carrying) and 4 in applicable_op:
      return 4
    return None
  
  def PRandom(self, agent, state, applicable_op):  # random policy
    if not applicable_op:
      return None
    carrying = bool(state[4])
    pref_action = self.preferred_action(carrying, applicable_op)
    if pref_action is not None:
      return pref_action
    return random.choice(applicable_op)  # otherwise choose an operator randomly

  def PExploit(self, agent, state, applicable_op):  # exploit policy
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
    eps = 1e-9
    best_ops = [a for a in applicable_op if abs(q_values[a] - best_value) <= eps]
    if random.random() < 0.8:
      return random.choice(best_ops)  # choose among the best, this is the tie break
    else:
      other = [a for a in applicable_op if a not in best_ops]
      # choose a different applicable op if possible
      return random.choice(other) if other else random.choice(applicable_op)

  def PGreedy(self, agent, state, applicable_op):  # greedy policy
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
      self.record_segment_step(agent, a, reward, s_next)
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
            print("*** Reached 6th terminal hit, experiment 4 stops ***")
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
      self.record_segment_step(agent, a, reward, s_next)
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
            print("*** Reached 6th terminal hit, experiment 4 stops ***")
            break

        world.reset()
        episode_steps = 0
        episode_rewards = 0
        episode_manhattan_dist = []
  
  def manhattan_distance(self):  # manhattan distance used for analysis to see how close/separated the agents were in execution
    return abs(self.world.F_pos[0] - self.world.M_pos[0]) + abs(self.world.F_pos[1] - self.world.M_pos[1])
  
  def record_segment_step(self, agent, action, reward, s_next):
    if agent == 'F':
      pos = (s_next[0], s_next[1])
      carry = bool(s_next[4])
    else:
      pos = (s_next[0], s_next[1])  # if the agent is M, we can use its view
      carry = bool(s_next[4])
    if action == 4 and reward > 0:  # start a new path right after a successful pickup
      self.seg_active[agent] = [pos]  # start at pickup cell
    if self.seg_active[agent] is not None and action in (0,1,2,3):  # if currently carrying append position on moves
      self.seg_active[agent].append(pos)
    if self.seg_active[agent] is not None and action == 5 and reward > 0:  # close the path on a successful dropoff
      path = tuple(self.seg_active[agent])
      if len(path) >= 1:
        self.segment_logs[agent].append(path)
      self.seg_active[agent] = None


