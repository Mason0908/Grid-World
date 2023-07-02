import numpy as np
from typing import Tuple, List, Union
from utils import Action, Move
from GridWorldTemplate import GridWorldTemplate


class GridWorld(GridWorldTemplate):

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
        super().__init__(world_height, world_width,
                         prob_direct, prob_lateral, rewards,
                         start_state, goal_states, walls=walls)

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
        def nextStateValid(state: np.array):
            if (state[0] >= 0 and
                state[0] <= (self.state_dim[0]-1) and
                state[1] >= 0 and
                state[1] <= (self.state_dim[1]-1) and
                    (tuple(state) not in self.walls)):
                return True
            return False

        tm = np.zeros((self.state_dim[0], self.state_dim[1],
                      self.action_dim, self.state_dim[0], self.state_dim[1]))
        for i in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                if (((i, j) not in self.walls) and ((i, j) not in self.goal_states)):
                    currState = np.array([i, j])
                    moveUp = Move.step(currState, Action.UP)
                    moveRight = Move.step(currState, Action.RIGHT)
                    moveDown = Move.step(currState, Action.DOWN)
                    moveLeft = Move.step(currState, Action.LEFT)

                    # move up
                    if (nextStateValid(moveUp)):
                        tm[i, j, Action.UP, i-1, j] += self.prob_direct
                        tm[i, j, Action.RIGHT, i-1, j] += self.prob_lateral
                        tm[i, j, Action.LEFT, i-1, j] += self.prob_lateral
                    else:
                        tm[i, j, Action.UP, i, j] += self.prob_direct
                        tm[i, j, Action.RIGHT, i, j] += self.prob_lateral
                        tm[i, j, Action.LEFT, i, j] += self.prob_lateral

                    # move right
                    if (nextStateValid(moveRight)):
                        tm[i, j, Action.RIGHT, i, j+1] += self.prob_direct
                        tm[i, j, Action.UP, i, j+1] += self.prob_lateral
                        tm[i, j, Action.DOWN, i, j+1] += self.prob_lateral
                    else:
                        tm[i, j, Action.RIGHT, i, j] += self.prob_direct
                        tm[i, j, Action.UP, i, j] += self.prob_lateral
                        tm[i, j, Action.DOWN, i, j] += self.prob_lateral

                    # move down
                    if (nextStateValid(moveDown)):
                        tm[i, j, Action.DOWN, i+1, j] += self.prob_direct
                        tm[i, j, Action.LEFT, i+1, j] += self.prob_lateral
                        tm[i, j, Action.RIGHT, i+1, j] += self.prob_lateral
                    else:
                        tm[i, j, Action.DOWN, i, j] += self.prob_direct
                        tm[i, j, Action.LEFT, i, j] += self.prob_lateral
                        tm[i, j, Action.RIGHT, i, j] += self.prob_lateral

                    # move left
                    if (nextStateValid(moveLeft)):
                        tm[i, j, Action.LEFT, i, j-1] += self.prob_direct
                        tm[i, j, Action.UP, i, j-1] += self.prob_lateral
                        tm[i, j, Action.DOWN, i, j-1] += self.prob_lateral
                    else:
                        tm[i, j, Action.LEFT, i, j] += self.prob_direct
                        tm[i, j, Action.UP, i, j] += self.prob_lateral
                        tm[i, j, Action.DOWN, i, j] += self.prob_lateral

        self.T = tm

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

        def nextStateValid(state: np.array):
            if (state[0] >= 0 and
                state[0] <= (self.state_dim[0]-1) and
                state[1] >= 0 and
                state[1] <= (self.state_dim[1]-1) and
                    (tuple(state) not in self.walls)):
                return True
            return False
        currState = np.array(state)
        nextState = np.array([-1, -1])
        if (action == Action.UP):
            options = np.array([Action.UP, Action.LEFT, Action.RIGHT])
            move = np.random.choice(
                options, p=[self.prob_direct, self.prob_lateral, self.prob_lateral])
            nextState = Move.step(currState, move)
        elif (action == Action.RIGHT):
            options = np.array([Action.RIGHT, Action.UP, Action.DOWN])
            move = np.random.choice(
                options, p=[self.prob_direct, self.prob_lateral, self.prob_lateral])
            nextState = Move.step(currState, move)
        elif (action == Action.DOWN):
            options = np.array([Action.DOWN, Action.LEFT, Action.RIGHT])
            move = np.random.choice(
                options, p=[self.prob_direct, self.prob_lateral, self.prob_lateral])
            nextState = Move.step(currState, move)
        elif (action == Action.LEFT):
            options = np.array([Action.LEFT, Action.UP, Action.DOWN])
            move = np.random.choice(
                options, p=[self.prob_direct, self.prob_lateral, self.prob_lateral])
            nextState = Move.step(currState, move)

        if nextStateValid(nextState):
            reward = self.R[nextState[0], nextState[1]]
            return tuple(nextState), reward
        else:
            reward = self.R[currState[0], currState[1]]
            return tuple(currState), reward
