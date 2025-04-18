from typing import Dict, List, Tuple
from src.gridworld import Gridworld
from src.value_iteration import perform_value_iteration, get_policy
from src.policy_iteration import policy_iteration
from src.visualization import Visualizer, display_convergence
import yaml
from src.utils import CONSOLE


def solve_maze(
    config_file: str,
) -> Tuple[List[Dict[int, float]], List[Dict[int, float]]]:
    # Load the configuration from the YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    size = config["size"]
    walls = config["walls"]
    # Change list into tuple
    walls = [tuple(wall) for wall in walls]
    terminal_states = config["terminal_states"]
    rewards = {
        tuple(map(int, key.strip("[]").split(","))): value
        for key, value in config["rewards"].items()
    }

    env = Gridworld(
        size,
        walls,
        terminal_states,
        rewards,
        config["transition_prob"],
        config["discount"],
        config["white_reward"],
    )

    visualizer = Visualizer(env=env)

    # Value iteration
    V_value_iter, value_iteration_log = perform_value_iteration(env)
    CONSOLE.print("Value Iteration:")
    CONSOLE.print(V_value_iter)

    # Get the optimal policy
    policy = get_policy(env, V_value_iter)
    CONSOLE.print("Policy:", policy)

    # Visualize the gridworld with the optimal policy
    visualizer.visualize_board()
    visualizer.visualize_policy(policy)
    visualizer.visualize_utilities(V_value_iter)

    display_convergence([value_iteration_log], ["Value Iteration"])

    # Policy iteration
    policy, V_policy_iter, policy_iteration_log = policy_iteration(env)
    CONSOLE.print("Policy Iteration:")
    CONSOLE.print("Policy:", policy)
    CONSOLE.print("Utilities:", V_policy_iter)

    # Visualize the gridworld with the optimal policy
    visualizer.visualize_policy(policy)
    visualizer.visualize_utilities(V_policy_iter)

    display_convergence([policy_iteration_log], ["Policy Iteration"])

    return value_iteration_log, policy_iteration_log
