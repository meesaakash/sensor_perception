"""
Run the simulation for a given scenario.
"""

import sys
from start_stop import start_stop
from intercepting_path import intercepting_path
from reinforcement_learning import rl
from experiment import run_random_experiment

error_message = "Please specify an algorithm to run.\n <algorithm>\n\t algorithm: start_stop, intercepting, rl"
ALGORITHMS = ["start_stop", "intercepting", "rl"]


def main():
    system_args = sys.argv[1:]
    plotting = False
    if len(system_args) > 2:
        print("Too many arguments.")
        print(error_message)
        return
    elif len(system_args) == 2:
        plotting = True
    
    if system_args[0] not in ALGORITHMS:
        print("Invalid algorithm.")
        print(error_message)
        return
    
    algorithm = system_args[0]

    if algorithm == "start_stop":
        print("Running start stop experiment...", "and I am plotting" if plotting else "")
        run_random_experiment(start_stop, plotting=plotting)
    elif algorithm == "intercepting":
        print("Running intercepting path experiment...", "and I am plotting" if plotting else "")
        run_random_experiment(intercepting_path, plotting=plotting)
    elif algorithm == "rl":
        print("Running reinforcement learning experiment...", "and I am plotting" if plotting else "")
        run_random_experiment(rl, plotting=plotting)

if __name__ == "__main__":
    main()