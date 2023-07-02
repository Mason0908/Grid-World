import numpy as np
from typing import Tuple, List, Union, Callable
from itertools import product
from utils import Action, Move

FLOAT_TOLERANCE = 1e-5


class GridWorldTemplate:
    WALL_TOKEN = "X"
    EMPTY_TOKEN = " "

    __doc__ = \
        """
        Template base class for a grid world environment where constructor and basic functions are defined here.
        
        Note: DO NOT CHANGE ANYTHING IN THIS CLASS!
        """

    def __init__(self, world_height: int, world_width: int,
                 prob_direct: float, prob_lateral: float, rewards: np.ndarray,
                 start_state: Tuple[int, int], goal_states: List[Tuple[int, int]],
                 walls: Union[None, List[Tuple[int, int]]] = None):
        """
        Constructor function.

        :param world_height: number of rows in world
        :param world_width: number of columns in world
        :param prob_direct: probability of going towards the intended direction
        :param prob_lateral: probability of going lateral to the intended direction
        :param rewards: a 2D Numpy array of rewards when entering each state
        :param start_state: state to start in
        :param goal_states: states that terminate the simulation
        :param walls: coordinates of states to be considered as walls
        """
        assert abs(prob_direct + prob_lateral * 2 - 1) < FLOAT_TOLERANCE, \
            f"The total probability must sum to 1 but the given probabilities sum to {prob_direct + prob_lateral * 2}."
        assert start_state not in goal_states, "We do not want trivial environments where you start at a goal state."
        assert len(goal_states) > 0, "There must be at least one goal state."

        self.state_dim = (world_height, world_width)
        self.action_dim = len(Action.space())

        self.start_state = start_state
        self.goal_states = goal_states
        self.walls = list() if walls is None else walls

        self.prob_direct = prob_direct
        self.prob_lateral = prob_lateral

        self.R = rewards
        self.T = None

    def view_transition(self, state: Tuple[int, int], action: int, verbose: bool = True) -> np.ndarray:
        """
        Displays the distribution of transition probabilities when starting at state and taking action.

        :param state: starting state
        :param action: action to take
        :param verbose: if True, prints additional information
        :return: human readable T(·|s, a)
        """
        if verbose:
            print("=" * 30)
            print(f"Starting at state {state} and taking action {Action.to_str(action)}:")
            printable_world = np.empty(self.state_dim, dtype=object)
            for s in product(range(self.state_dim[0]), range(self.state_dim[1])):
                if s == state:
                    printable_world[s] = Action.to_token(action)
                elif s not in self.walls:
                    printable_world[s] = GridWorldTemplate.EMPTY_TOKEN
                else:
                    printable_world[s] = GridWorldTemplate.WALL_TOKEN

            print(printable_world)

        printable_T = self.T[state[0], state[1], action, ...].astype(object)

        for w1, w2 in self.walls:
            printable_T[w1, w2] = GridWorldTemplate.WALL_TOKEN

        return printable_T

    def fill_T(self) -> None:
        """
        Initializes and populates transition probabilities T(s'|s,a) for all possible (s, a, s').
        The computed transition probabilities are stored in self.T.

        Usage:
        >> env = GridWorld(2, 3, ...)
        >> print(env.T)  # None
        >> env.fill_T()
        >> print(env.T.shape)
        (2, 3, 4, 2, 3)
        """
        raise NotImplementedError

    def make_move(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float]:
        """
        Takes a single step in the environment under stochasticity based on the transition probabilities.

        Usage:
        >> env  # created; prob_direct = 0.5, prob_lateral = 0.25
        >> env.make_move((0, 0), Action.DOWN)
        ((1, 0), -0.04)
        >> env.make_move((0, 0), Action.DOWN)  # same start state but moved laterally because of randomness
        ((0, 1), -0.04)

        :param state: starting state
        :param action: action to taken
        :return: next state and observed reward entering the next state
        """
        raise NotImplementedError

    def simulate(self) -> Tuple[Tuple[int, int],
                                Callable[[int], Tuple[int, Tuple[int, int], float, bool]]]:
        """
        Enables simulated environment which starts at the start state and at each time step,
        an action can be taken which probabilistically takes you to a next state and grants a reward.
        Whenever you land on a goal state, a "done" flag will be set to True and you should not
        attempt to take another action.

        Usage:
        >> cur_state, callback_fn = env.simulate()
        >> print(cur_state)
        (0, 0)
        >> i, next_state, r, done = callback_fn(Action.DOWN); print(i, next_state, r, done)
        1 (1, 0) -0.04 False
        >> i, next_state, r, done = callback_fn(Action.DOWN); print(i, next_state, r, done)
        2 (2, 0) -0.04 False

        >> ...  # further executions of similar pattern

        >> i, next_state, r, done = callback_fn(Action.RIGHT); print(i, next_state, r, done)
        5 (2, 3) 1.0 True

        Note: This function requires GridWorld.make_move to be implemented correctly.

        :return: initial start state and a callback function that you are to call to take an action in
            the environment which returns a tuple containing 4 items:
            1) current time step
            2) next state
            3) observed reward entering the next state
            4) "done" flag (which is always False except at the end of simulation)
        """
        cur_state = [self.start_state]  # mutable reference
        i = [0]  # mutable reference

        def callback_fn(action):
            if cur_state[0] in self.goal_states:
                raise RuntimeError("Goal state has been reached, but an attempt was made to take an action.")

            next_state, reward = self.make_move(cur_state[0], action)
            cur_state[0] = next_state
            done = (cur_state[0] in self.goal_states)
            i[0] += 1
            return i[0], next_state, reward, done

        return self.start_state, callback_fn
