import numpy as np
from typing import List, Tuple


# Note: DO NOT CHANGE ANYTHING IN THIS FILE!


class Action:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    UP_TOKEN = "↑"
    RIGHT_TOKEN = "→"
    DOWN_TOKEN = "↓"
    LEFT_TOKEN = "←"

    @classmethod
    def space(cls) -> List[int]:
        """
        :return: a list of all actions possible (in a clockwise manner starting from UP)
        """
        return [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]

    @classmethod
    def to_str(cls, action: int) -> str:
        """
        Converts an action to its corresponding word string.

        :param action: action to be converted into its name
        :return: action name
        """
        if action == cls.UP:
            return "UP"
        elif action == cls.RIGHT:
            return "RIGHT"
        elif action == cls.DOWN:
            return "DOWN"
        elif action == cls.LEFT:
            return "LEFT"
        else:
            raise ValueError(f"{action} is not a valid action")

    @classmethod
    def to_token(cls, action: int) -> str:
        """
        Convert an action to its corresponding arrow string.

        :param action: action to be converted into arrow string
        :return: an arrow for the action
        """
        if action == cls.UP:
            return cls.UP_TOKEN
        elif action == cls.RIGHT:
            return cls.RIGHT_TOKEN
        elif action == cls.DOWN:
            return cls.DOWN_TOKEN
        elif action == cls.LEFT:
            return cls.LEFT_TOKEN
        else:
            raise ValueError(f"{action} is not a valid action")


class Move:
    __doc__ = \
        """
        Provides a simpler interface to update a state given an action.
        Requires the state to be in a NumPy represetation.
        
        Usage:
        >> cur_state = np.array([0, 1])
        >> action = Action.RIGHT
        >> print(Move.step(cur_state, action))  # preferred, but not required
        [0 2]
        >> move = Move.from_action(action)  # alternative to above
        >> print(cur_state + move)
        [0 2]
        """

    UP = np.array([-1, 0])
    RIGHT = np.array([0, 1])
    DOWN = np.array([1, 0])
    LEFT = np.array([0, -1])

    @classmethod
    def step(cls, state: np.ndarray, action: int) -> np.ndarray:
        """
        Computes the resultant state when starting at some state and taking some action.

        :param state: starting state
        :param action: action to take
        :return: resultant state
        """
        if action == Action.UP:
            return state + cls.UP
        elif action == Action.RIGHT:
            return state + cls.RIGHT
        elif action == Action.DOWN:
            return state + cls.DOWN
        elif action == Action.LEFT:
            return state + cls.LEFT
        else:
            raise ValueError(f"{action} is not a valid action.")

    @classmethod
    def from_action(cls, action: int) -> np.ndarray:
        """
        :param action: action to convert
        :return: converts an action to its corresponding move
        """
        if action == Action.UP:
            return cls.UP
        elif action == Action.RIGHT:
            return cls.RIGHT
        elif action == Action.DOWN:
            return cls.DOWN
        elif action == Action.LEFT:
            return cls.LEFT
        else:
            raise ValueError(f"{action} is not a valid action.")
