import numpy as np
from typing import Tuple, Union, List, TypeVar
from utils import Action, Move

GridWorld_Type = TypeVar("GridWorld")


class AgentTemplate:
    WALL_VALUE = 0.
    WALL_TOKEN = " "
    GOAL_TOKEN = "O"

    __doc__ = \
        """
        Template base class for an agent where constructor and basic functions are defined here.
        
        Note: DO NOT CHANGE ANYTHING IN THIS CLASS!
        """

    def __init__(self, env: GridWorld_Type):
        """
        Constructor function.

        :param env: environment to put the agent in
        """
        self.env = env

    def view_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Displays a grid world representation of a given policy.

        :param policy: policy to visualize
        :return: formatted policy using arrows (walls will be empty, goals will be O)
        """
        env = self.env
        printable_policy = np.vectorize(Action.to_token)(policy)

        for w1, w2 in env.walls:
            printable_policy[w1, w2] = AgentTemplate.WALL_TOKEN

        for g1, g2 in env.goal_states:
            printable_policy[g1, g2] = AgentTemplate.GOAL_TOKEN

        return printable_policy

    def get_path(self, policy: np.ndarray, start_state: Tuple[int, int],
                 goal_states: List[Tuple[int, int]], max_iter: int = np.inf) -> List[str]:
        """
        Determines the path from the start state to the goal state following the optimal policy.

        Usage:
        >> print(agent.view_policy(pi))
        [['↓' '↓']
         ['→' 'O']]
        >> print(agent.get_path(pi, (0, 0), [(1, 1)]))
        ['DOWN', 'RIGHT']

        :param policy: policy array
        :param start_state: starting state based on the grid world
        :param goal_states: states that terminate the simulation based on the grid world
        :param max_iter: maximum allowed iterations
        :return: list of actions agent should take to get from start_state to the goal_state
        """
        cur_state = np.array(start_state)
        path = list()
        i = 0

        while tuple(cur_state) not in goal_states and i < max_iter:
            action = policy[tuple(cur_state)]
            path.append(action)
            cur_state = Move.step(cur_state, action)
            i += 1

        return list(map(Action.to_str, path))

    def value_iteration(self, gamma: float, tolerance: float = 0.001,
                        max_iter: int = np.inf) -> Tuple[np.ndarray, int]:
        """
        Performs value iteration on the loaded environment. The tolerance parameter specifies the
        maximum absolute difference of all entries between the current V and the previous V
        before the iteration can stop. The max_iter parameter is a hard stopping criterion
        where the iteration must stop once the maximum number of iterations is reached. The tolerance
        and max_iter parameter should be used as a conjunction.

        Usage:
        >> V, i = agent.value_iteration(0.99)

        :param gamma: discount factor
        :param tolerance: terminating condition for the iteration loop
        :param max_iter: maximum allowable iterations (should be set to np.inf unless you're getting infinite loops)
        :return: optimal value array and number of iterations
        """
        raise NotImplementedError

    def find_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Finds the best action to take at each state based on the state values obtained,
        i.e. computing pi(s) = argmax_a(sum_{s'}(P(s'|s, a) V(s'))) for all s

        :param V: state value array to extract the policy from
        :return: policy array
        """
        raise NotImplementedError

    def passive_adp(self, policy: np.ndarray,
                    gamma: float, adp_iters: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Performs passive ADP on a given policy using simulations from the environment.

        Note:
        * You should be using the GridWorld.simulate method which requires the GridWorld.make_move
        to be fully implemented and working.
        * You are not allowed to use the true transition probabilities from the environment (i.e. env.T) but
        instead you should learn an estimate of the transition probabilities through experiences
        from GridWorld.simulate.

        Usage:
        >> V, _ = agent.value_iteration(0.99)
        >> pi = agent.find_policy(V)
        >> V_adp, all_Vs = agent.passive_adp(pi, 0.99, adp_iters=1000)
        >> print(len(all_Vs))
        1000

        :param policy: policy to determine action to take for each state
        :param gamma: discount factor
        :param adp_iters: number of the passive ADP iterations
        :return: state value array (V) based on the given policy, and a list of V's per iteration
        """
        raise NotImplementedError
