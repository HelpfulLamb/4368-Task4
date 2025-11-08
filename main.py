import argparse
from experiments import ExperimentRunner
from two_agent_sys import TwoAgentRLSystem
from visuals import Visualizer

def run_exp_1(variant: str, runs=(1,2), alpha=0.3, gamma=0.5):
  assert variant in {"a", "b", "c"}, "Experiment 1 variant must be a, b, or c"
  rl_runner = ExperimentRunner()
  plot = Visualizer()
  all_results = []
  if variant == "a":
    policy = "PRANDOM"
  elif variant == "b":
    policy = "PGREEDY"
  else:
    policy = "PEXPLOIT"
  print(f"Experiment 1: Q-Learning with Different Policies: {policy}\n--- Learning Rate = {alpha}, Discount Factor = {gamma} ---")
  for run in runs:
    print(f"\n--- Run {run} ---")
    rl_runner.system = TwoAgentRLSystem()
    result = rl_runner.experiment_1(variant, seed=10+run, alpha=alpha, gamma=gamma)
    all_results.append(result)

    # optional plot
    try:
      plot.plot_learning_curve(rl_runner.system.performance_history, 'F', run=run)
      plot.plot_steps_per_episode(rl_runner.system.performance_history, 'F', run=run)
    except Exception as e:
      print(f"[plot warning]: {e}")

    rl_runner.analyze_learned_paths(result)
    print(f"\nQ-Table for Experiment 1.{variant}: Run {run}")
    rl_runner.print_q_table('F', sort_by_best=True)  # change to M for other agent
  return all_results

def run_exp_2(runs=(1,2), alpha=0.3, gamma=0.5):
  rl_runner = ExperimentRunner()
  plot = Visualizer()
  all_results = []
  print(f"Experiment 2: SARSA Learning\n--- Learning Rate = {alpha}, Discount Factor = {gamma} ---")
  for run in runs:
    print(f"\n--- Run {run} ---")
    rl_runner.system = TwoAgentRLSystem()
    result = rl_runner.experiment_2(seed=10+run, alpha=alpha, gamma=gamma)
    all_results.append(result)

    # optional plot
    try:
      plot.plot_learning_curve(rl_runner.system.performance_history, 'F', run=run)
      plot.plot_steps_per_episode(rl_runner.system.performance_history, 'F', run=run)
    except Exception as e:
      print(f"[plot warning]: {e}")

    rl_runner.analyze_learned_paths(result)
    print(f"\nQ-Table for Experiment 2: Run {run}")
    rl_runner.print_q_table('F', sort_by_best=True)  # or M
  return all_results

def run_exp_3(runs=(1,2), learning_rates = (0.15, 0.45), gamma = 0.5, learning="q"):
  rl_runner = ExperimentRunner()
  plot = Visualizer()
  all_results = []
  if learning == 'q':
    rl_type = "Q-Learn"
  else:
    rl_type = "SARSA"
  print(f"Experiment 3: Tuning Learning Rates for {rl_type}")
  for lr in learning_rates:  # 2 runs for each diff learning rate
    for run in runs:
      print(f"\n--- Run {run} ---")
      rl_runner.system = TwoAgentRLSystem()
      result = rl_runner.experiment_3(seed=10+run, alpha=lr, gamma=gamma, learn_type=learning)
      all_results.append(result)

      # optional plot
      try:
        plot.plot_learning_curve(rl_runner.system.performance_history, 'F', run=run)
        plot.plot_steps_per_episode(rl_runner.system.performance_history, 'F', run=run)
      except Exception as e:
        print(f"[plot warning]: {e}")
      
      rl_runner.analyze_learned_paths(result)
      print(f"\nQ-Table for Experiment 3: Run {run}")
      rl_runner.print_q_table('F', sort_by_best=True)  # or M
  return all_results

def run_exp_4(runs=(1,2), alpha=0.3, gamma=0.5, learning="s"):
  rl_runner = ExperimentRunner()
  plot = Visualizer()
  all_results = []
  if learning == 's':
    rl_type = "SARSA"
  else:
    rl_type = "Q-Learn"
  print(f"Experiment 4: Adaptation to Changes in Pickup Locations for {rl_type}\n--- Learning Rate = {alpha}, Discount Factor = {gamma} ---")
  for run in runs:
    print(f"\n--- Run {run} ---")
    rl_runner.system = TwoAgentRLSystem()
    result = rl_runner.experiment_4(seed=10+run, alpha=alpha, gamma=gamma, learn_type=learning)
    all_results.append(result)

    try:
      plot.plot_learning_curve(rl_runner.system.performance_history, 'F', run=run)
      plot.plot_steps_per_episode(rl_runner.system.performance_history, 'F', run=run)
    except Exception as e:
      print(f"[plot warning]: {e}")

    rl_runner.analyze_learned_paths(result)
    print(f"\nQ-Table for Experiment 4: Run {run}")
    rl_runner.print_q_table('F', sort_by_best=True)  # or M
  return all_results

def parse_args():
  parser = argparse.ArgumentParser(
    description="Run specific RL experiments: 1a, 1b, 1c, 2, 3q, 3s, 4q, 4s"
  )
  parser.add_argument("which", choices=["1a", "1b", "1c", "2", "3q", "3s", "4q", "4s"], help="Experiment to run")
  parser.add_argument("--runs", type=int, nargs="+", default=[1,2], help="Run ids to execute")
  parser.add_argument("--alpha", type=float, default=0.3, help="Learning rate for experiments 1, 2, and 4")
  parser.add_argument("--lrs", type=float, nargs="+", default=[0.15, 0.45])
  parser.add_argument("--gamma", type=float, default=0.5, help="Discount factor")
  return parser.parse_args()

def main():
  args = parse_args()
  which = args.which
  runs = tuple(args.runs)
  if which.startswith("1"):
    variant = which[-1]
    run_exp_1(variant, runs=runs, alpha=args.alpha, gamma=args.gamma)
  elif which.startswith("2"):
    run_exp_2(runs=runs, alpha=args.alpha, gamma=args.gamma)
  elif which.startswith("3"):
    rl_type = which[-1]
    run_exp_3(runs=runs, learning_rates=tuple(args.lrs), gamma=args.gamma, learning=rl_type)
  elif which.startswith("4"):
    rl_type = which[-1]
    run_exp_4(runs=runs, alpha=args.alpha, gamma=args.gamma, learning=rl_type)
  else:
    raise ValueError(f"Unknown experiment '{which}'")

# run the program
if __name__ == "__main__":
  main()
