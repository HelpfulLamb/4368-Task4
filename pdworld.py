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