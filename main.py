from experiments import ExperimentRunner
from two_agent_sys import TwoAgentRLSystem

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
