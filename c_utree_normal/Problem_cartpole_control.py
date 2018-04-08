# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/Problem_cartpole_control.py
# Compiled at: 2017-12-04 13:56:08
from datetime import datetime
import gym


class CartPole:
    """
    An MDP. Contains methods for initialisation, state transition.
    Can be aggregated or unaggregated.
    """

    def __init__(self, games_directory='../save_all_transition/', gamma=1):
        assert games_directory is not None
        self.games_directory = games_directory
        self.actions = {'left': 0,
                        'right': 1
                        }
        self.stateFeatures = {'Cart_Position': 'continuous', '	Cart_Velocity': 'continuous', 'Pole_Angle': 'continuous',
                              'Pole_Velocity_At_Tip	': 'continuous'
                              }
        self.gamma = gamma
        self.reset = None
        self.isEpisodic = True
        self.nStates = len(self.stateFeatures)
        self.dimNames = ['Cart_Position', 'Cart_Velocity', 'Pole_Angle', 'Pole_Velocity_At_Tip']
        self.dimSizes = ['continuous', 'continuous', 'continuous', 'continuous']
        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        self.probName = ('{0}_gamma={1}_mode={2}').format(d, gamma,
                                                          'Action Feature States' if self.nStates > 12 else 'Feature States')
        # self.games_directory = '/cs/oschulte/Galen/Hockey-data-entire/State-Hockey-Training-All-feature5-scale-neg_reward_v_correct_/'
        # self.games_directory = '../save_all_transition/'
        self.env = gym.make('CartPole-v0')
        return
