from typing import Dict, List, Tuple
import numpy as np
from .gridworld import Gridworld
from tensorboardX import SummaryWriter
from src.utils import CONSOLE


def get_policy(env: Gridworld, utilities: np.ndarray) -> np.ndarray:
    """
    Derive the optimal policy based on the computed utilities.

    Args:
        env (Gridworld): The gridworld environment.
        utilities (np.ndarray): The utility values for all states.

    Returns:
        np.ndarray: The optimal policy for the environment.
    """
    policy = np.zeros(env.size, dtype=object)
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for row in range(env.size[0]):
        for col in range(env.size[1]):
            if (
                (row, col)in env.walls
                or (row, col)in env.terminal_states
                or (row, col)in env.rewards.keys()
            ):
                continue
            q_values = [
                sum(
                    prob
                    * (
                        env.get_reward((row, col), action, next_state)
                        + env.discount * utilities[next_state]
                    )
                    for next_state, prob in env.get_transition_states_and_probs(
                        (row, col), action
                    )
                )
                for action in env.actions
            ]
            policy[row, col] = actions[np.argmax(q_values)]
    return policy


def perform_value_iteration(
    env: Gridworld, threshold: float = 0.001, min_iteration: int = 50
) -> Tuple[np.ndarray, List[Dict[int, float]]]:
    """
    Execute the value iteration algorithm to compute optimal utility values.

    Args:
        env (Gridworld): The gridworld environment.
        threshold (float): Convergence threshold for utility updates. Default is 0.001.
        min_iterations (int): Minimum number of iterations to perform. Default is 50.

    Returns:
        Tuple[np.ndarray, List[Dict[int, float]]]: 
        The optimal utility values and a log of utility values over iterations.
    """
    writer = SummaryWriter("runs/maze_solver_experiment")
    V = np.zeros(env.size)
    log = []
    iteration = 0
    while True:
        delta = 0
        for row in range(env.size[0]):
            for col in range(env.size[1]):
                if (
                    (row, col)in env.walls
                    or (row, col)in env.terminal_states
                    or (row, col)in env.rewards.keys()
                ):
                    continue
                v = V[row, col]
                V[row, col] = max(
                    [
                        sum(
                            prob
                            * (
                                env.get_reward((row, col), action, next_state)
                                + env.discount * V[next_state]
                            )
                            for next_state, prob in env.get_transition_states_and_probs(
                                (row, col), action
                            )
                        )
                        for action in env.actions
                    ]
                )
                delta = max(delta, abs(v - V[row, col]))

        writer.add_scalar("Value Iteration Delta", delta, iteration)
        writer.add_scalar("Value Iteration Utilities", V.mean(), iteration)
        log.append({iteration: V.mean()})
        iteration += 1
        if iteration == min_iteration:
            if delta < threshold:
                CONSOLE.print("Value iteration converged", style="bold green")
            else:
                CONSOLE.print("Value iteration did not converge", style="bold red")
            break
    return V, log
