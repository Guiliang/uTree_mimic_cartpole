import optparse
import Problem_cartpole
import pickle
import Agent_boost_Galen as Agent

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 10000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=200,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")
optparser.add_option("-g", "--game number to train", dest="GAME_NUMBER", default=None,
                     help="which game to train")
optparser.add_option("-e", "--training mode", dest="TRAINING_MODE", default='_linear_epoch_decay_lr',
                     help="training mode")

opts = optparser.parse_args()[0]


def train():
    mc_problem = Problem_cartpole.CartPole(games_directory=opts.GAME_DIRECTORY)
    CUTreeAgent = Agent.CUTreeAgent(problem=mc_problem, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0, training_mode=opts.TRAINING_MODE)

    # CUTreeAgent.add_linear_regression()
    if opts.GAME_NUMBER is None:
        CUTreeAgent.episode(game_number=0)
    else:
        CUTreeAgent.episode(game_number=int(opts.GAME_NUMBER))


def one_shot_train():
    mc_problem = Problem_cartpole.CartPole(games_directory=opts.GAME_DIRECTORY)
    CUTreeAgent = Agent.CUTreeAgent(problem=mc_problem, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0, training_mode=opts.TRAINING_MODE)
    CUTreeAgent.one_shot_episode()


def test_sh():
    print "hello"


if __name__ == "__main__":
    train()
    # one_shot_train()
