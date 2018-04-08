import random

import Agent_boost_Galen as Agent
import Problem_cartpole
import C_UTree_boost_Galen as C_UTree
import gym
import numpy as np
import tensorflow as tf
import linear_regression
import gc
import scipy.stats
import scipy as sp

env = gym.make('CartPole-v0')
ACTION_LIST = [0, 1]


def get_action_linear_regression(observation, CUTreeAgent):
    Q_list = []
    Q_number = []
    for action_test in ACTION_LIST:
        sess = tf.Session()
        inst = C_UTree.Instance(-1, observation, action_test, observation, None,
                                None)  # leaf is located by the current observation
        node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)
        LR = linear_regression.LinearRegression()
        LR.read_weights(weights=node.weight, bias=node.bias)
        LR.readout_linear_regression_model()
        sess.run(LR.init)
        temp = sess.run(LR.pred, feed_dict={LR.X: [inst.currentObs]}).tolist()
        Q_list.append(temp)
        Q_number.append(len(node.instances))
    return ACTION_LIST[Q_list.index(max(Q_list))]


def get_action_similar_instance(observation, CUTreeAgent):
    mse_criterion = 0.00015
    done = False
    # for criterion_index in range(0, len(mse_criterion)):
    while done is not True:

        # action = None
        top_actions = []
        min_mse = 10000
        Q_value = 0

        length = 0

        for action_test in ACTION_LIST:
            inst = C_UTree.Instance(-1, observation, action_test, observation, None,
                                    None)  # leaf is located by the current observation
            node = CUTreeAgent.utree.getAbsInstanceLeaf(inst)

            length = len(node.instances)

            for instance in node.instances:
                instance_observation = instance.currentObs
                mse = ((np.asarray(observation) - np.asarray(instance_observation)) ** 2).mean()
                if mse < min_mse:
                    min_mse = mse
                    action_min = action_test
                if mse < mse_criterion:
                    top_actions.append(action_test)

        if len(top_actions) >= 3:
            done = True
            a = np.asarray(top_actions)
            counts = np.bincount(a)
            action_most = np.argmax(counts)
            # if action != action_most:
            # print 'catch you'
            action = action_most
        elif mse_criterion >= 0.0005:
            done = True
            action = action_min

        if done:
            break
        else:
            # print 'append'
            mse_criterion += 0.00005

        gc.collect()

    return action


def test():
    ice_hockey_problem = Problem_cartpole.CartPole(games_directory='../save_all_transition/')
    CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='_linear_epoch_decay_lr')
    CUTreeAgent.read_Utree(game_number=165, save_path=CUTreeAgent.SAVE_PATH)

    reward_list = []
    for i in range(100):
        observation = env.reset()
        done = False
        count = 0
        total_reward = 0

        while not done:
            env.render()

            # action = get_action_similar_instance(observation.tolist(), CUTreeAgent)
            action = 1
            newObservation, reward, done, _ = env.step(action)

            observation = newObservation
            total_reward += reward
            count += 1
            # print('U-tree: The episode ' + str(i) + ' lasted for ' + str(
            #     count) + ' time steps' + ' with action ' + str(action))
        print ' lasted for ' + str(count)
        reward_list.append(total_reward)

    mean, var, h = mean_confidence_interval(reward_list)
    print 'mean:{0}, variance:{2}, +-{1}'.format(str(mean), str(h), str(var))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    var = np.var(a)
    m, sd, se = np.mean(a), np.var(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, var, h


if __name__ == "__main__":
    test()
