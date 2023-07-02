import numpy as np
from typing import Tuple, Union, List, TypeVar
from utils import Action, Move
from AgentTemplate import AgentTemplate

GridWorld_Type = TypeVar("GridWorld")


class Agent(AgentTemplate):

    def __init__(self, env: GridWorld_Type):
        """
        Constructor function.

        :param env: environment to put the agent in
        """
        super().__init__(env)

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

        V = np.zeros((self.env.state_dim[0], self.env.state_dim[1]))
        for s in self.env.goal_states:
            V[s[0], s[1]] = self.env.R[s[0], s[1]]

        if (self.env.T is None):
            self.env.fill_T()
        T = self.env.T
        count = 0
        while (count < max_iter):
            VTemp = np.copy(V)
            maxDiff = 0

            for i in range(self.env.state_dim[0]):
                for j in range(self.env.state_dim[1]):
                    if (((i, j) not in self.env.walls) and ((i, j) not in self.env.goal_states)):
                        maxUtility = float('-inf')
                        for a in Action.space():
                            tempValue = 0
                            currT = T[i, j, a]
                            for k in range(len(currT)):
                                for m in range(len(currT[0])):
                                    tempValue += currT[k, m] * V[k, m]
                            if (tempValue > maxUtility):
                                maxUtility = tempValue
                        VTemp[i, j] = self.env.R[i, j] + gamma*maxUtility
                        if (abs(V[i, j]-VTemp[i, j]) > maxDiff):
                            maxDiff = abs(V[i, j]-VTemp[i, j])

            V = np.copy(VTemp)
            if (maxDiff <= tolerance):
                break
            count += 1

        return V, count

    def find_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Finds the best action to take at each state based on the state values obtained,
        i.e. computing pi(s) = argmax_a(sum_{s'}(P(s'|s, a) V(s'))) for all s

        :param V: state value array to extract the policy from
        :return: policy array
        """

        policy = np.zeros(
            (self.env.state_dim[0], self.env.state_dim[1]), dtype=np.int8)
        if (self.env.T is None):
            self.env.fill_T()
        T = self.env.T

        for i in range(self.env.state_dim[0]):
            for j in range(self.env.state_dim[1]):
                values = np.zeros((4))
                for a in Action.space():
                    tempValue = 0
                    currT = T[i, j, a]
                    for k in range(len(currT)):
                        for m in range(len(currT[0])):
                            tempValue += currT[k, m] * V[k, m]
                    values[a] = tempValue
                policy[i, j] = np.argmax(values)

        return policy

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
        >> V_adp, all_Vs = agent.passive_adp(pi, 0.99)

        :param policy: policy to determine action to take for each state
        :param gamma: discount factor
        :param adp_iters: number of the passive ADP iterations
        :return: state value array based on the given policy, and all Vs per iteration
        """

        def solve_V_value_iteration(R: np.array, T: np.array, V_init: np.array, tolerance: float = 0.001,
                                    max_iter: int = np.inf):
            # V = np.zeros((self.env.state_dim[0], self.env.state_dim[1]))
            V = np.copy(V_init)
            for s in self.env.goal_states:
                V[s[0], s[1]] = R[s[0], s[1]]
            # count = 0
            # while (count < max_iter):
            #     VTemp = np.copy(V)
            #     maxDiff = 0

            #     for i in range(self.env.state_dim[0]):
            #         for j in range(self.env.state_dim[1]):
            #             if (((i, j) not in self.env.walls) and ((i, j) not in self.env.goal_states)):

            #                 tempValue = 0
            #                 a = policy[i, j]
            #                 currT = T[i, j, a]
            #                 for k in range(len(currT)):
            #                     for m in range(len(currT[0])):
            #                         tempValue += currT[k, m] * V[k, m]

            #                 VTemp[i, j] = R[i, j] + gamma*tempValue
            #                 if (abs(V[i, j]-VTemp[i, j]) > maxDiff):
            #                     maxDiff = abs(V[i, j]-VTemp[i, j])

            #     V = np.copy(VTemp)
            #     if (maxDiff <= tolerance):
            #         break
            #     count += 1
            VTemp = np.copy(V)

            for i in range(self.env.state_dim[0]):
                for j in range(self.env.state_dim[1]):
                    if (((i, j) not in self.env.walls) and ((i, j) not in self.env.goal_states)):

                        tempValue = 0
                        a = policy[i, j]
                        currT = T[i, j, a]
                        for k in range(len(currT)):
                            for m in range(len(currT[0])):
                                tempValue += currT[k, m] * VTemp[k, m]

                        VTemp[i, j] = R[i, j] + gamma*tempValue
                        

            

            return VTemp

        totalN = np.zeros(
            (self.env.state_dim[0], self.env.state_dim[1], self.env.action_dim))
        N = np.zeros((self.env.state_dim[0], self.env.state_dim[1],
                     self.env.action_dim, self.env.state_dim[0], self.env.state_dim[1]))
        T = np.zeros((self.env.state_dim[0], self.env.state_dim[1],
                     self.env.action_dim, self.env.state_dim[0], self.env.state_dim[1]))

        reward = np.zeros((self.env.state_dim[0], self.env.state_dim[1]))
        V = np.zeros((self.env.state_dim[0], self.env.state_dim[1]))
        Vs = list()
        Vs.append(V)
        count = 0
        currState, simulate = self.env.simulate()

        while (count < adp_iters):
            done = False
            currState, simulate = self.env.simulate()
            while (not done):
                action = int(policy[currState[0], currState[1]])
                i, nextState, r, done = simulate(action)
                reward[nextState[0], nextState[1]] = r
                totalN[currState[0], currState[1], action] += 1
                N[currState[0], currState[1], action,
                    nextState[0], nextState[1]] += 1

                for k in range(self.env.state_dim[0]):
                    for j in range(self.env.state_dim[1]):

                        T[currState[0], currState[1], action, k, j] = N[currState[0],
                                                                        currState[1], action, k, j] / totalN[currState[0], currState[1], action]

                currState = nextState
                V = solve_V_value_iteration(reward, T, V)
            Vs.append(V)
            # print(len(Vs))
            count += 1
        
        return V, Vs
