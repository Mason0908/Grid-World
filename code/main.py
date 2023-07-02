import numpy as np
from matplotlib import pyplot as plt

from GridWorld import GridWorld
from Agent import Agent
from utils import Action
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    np.random.seed(486)

    rewards = np.array([
        [-0.04, -0.04, -0.04, -0.04],
        [-0.04, -0.04, -0.04, -1.00],
        [-0.04, -0.04, -0.04, 1.000]
    ])

    start_state = (0, 0)
    goal_states = [(2, 3), (1, 3)]
    # walls = [(1, 1)]
    walls = [(1, 1), (0, 2)]

    env = GridWorld(world_height=3, world_width=4,
                    prob_direct=0.8, prob_lateral=0.1,
                    rewards=rewards,
                    start_state=start_state,
                    goal_states=goal_states,
                    walls=walls)
    env.fill_T()

    agent = Agent(env)

    V, i = agent.value_iteration(0.99)
    p = agent.find_policy(V)
    V_adp, Vs = agent.passive_adp(p, 0.99, 2000)
    # print(len(Vs))
    # x = np.arange(0, 2001)
    # y1 = np.zeros((2001))
    # y2 = np.zeros((2001))
    # y3 = np.zeros((2001))
    # y4 = np.zeros((2001))
    # for i in range(len(Vs)):
    #     y1[i] = Vs[i][0,0]
    #     y2[i] = Vs[i][0,3]
    #     y3[i] = Vs[i][2,2]
    #     y4[i] = Vs[i][1,2]
        
    # plt.title("State") 
    # plt.xlabel("iteration") 
    # plt.ylabel("V") 
    # plt.plot(x,y1,color='tab:blue',label='(0,0)') 
    # plt.plot(x,y2,color='tab:red',label='(0,3)') 
    # plt.plot(x,y3,color='tab:green',label='(2,2)') 
    # plt.plot(x,y4,color='tab:orange',label='(1,2)') 
    # plt.legend()
    # plt.show()
    print(V)
    print(V_adp)
    # print(p)
    # print(agent.get_path(p, (0, 0), goal_states=goal_states))
    # print(agent.view_policy(p))

    # print(env.make_move((0, 0), Action.LEFT))

#     rewards = np.array([
#         [-0.04, -0.04, -0.04],
#         [-0.04, -1.00, 1.000],
#         [-0.04, -0.04, -0.04]
#     ])

#     start_state = (0, 0)
#     goal_states = [(1, 1), (1, 2)]
#     walls = [(0, 1)]
#     env = GridWorld(world_height=3, world_width=3,
#                     prob_direct=0.8, prob_lateral=0.1,
#                     rewards=rewards,
#                     start_state=start_state,
#                     goal_states=goal_states,
#                     walls=walls)

#     # env.fill_T()
#     agent = Agent(env)
# #     V = np.array([[ 0.27989476,  0. ,         0.93765586],
# #  [ 0.3343869 , -1.     ,     1.        ],
# #  [ 0.55603969 , 0.64132247 , 0.90509417]])

#     V, i = agent.value_iteration(0.99)
#     p = agent.find_policy(V)

#     # print(p)

#     V_adp, Vs = agent.passive_adp(p, 0.99, 1000)
#     print(V_adp)
