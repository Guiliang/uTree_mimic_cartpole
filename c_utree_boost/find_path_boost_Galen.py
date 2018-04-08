import optparse
import Problem_cartpole
import pickle
import Agent_boost_Galen as Agent
import C_UTree_boost_Galen as C_UTree

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 10000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=1200,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")
optparser.add_option("-g", "--game number to test", dest="GAME_NUMBER", default=100,
                     help="which game to test")
optparser.add_option("-a", "--result correlation dir", dest="SAVE_CORRELATION_DIR", default=None,
                     help="the dir correlation result")
optparser.add_option("-j", "--result relative absolute error dir", dest="SAVE_RAE_DIR", default=None,
                     help="the dir relative absolute error result")
optparser.add_option("-i", "--result relative square error dir", dest="SAVE_RSE_DIR", default=None,
                     help="the dir relative square error result")
optparser.add_option("-b", "--result mean square error dir", dest="SAVE_MSE_DIR", default=None,
                     help="the dir mean square error result")
optparser.add_option("-f", "--result mean absolute error dir", dest="SAVE_MAE_DIR", default=None,
                     help="the dir mean absolute error result")
optparser.add_option("-e", "--training mode", dest="TRAINING_MODE", default='_linear_epoch_decay_lr',
                     help="training mode")

opts = optparser.parse_args()[0]


def recursive_find_path(node, target_idx):
    current_node_idx = node.idx

    if current_node_idx == target_idx:
        return True, "Q value = {0}, weight = {1} ".format(node.qValues, node.weight)
    else:
        for c_index in range(0, len(node.children)):
            child = node.children[c_index]
            find_flag, path = recursive_find_path(child, target_idx)

            if find_flag:
                if c_index == 0:
                    mark = '<'
                else:
                    mark = '>'
                if node.distinction.continuous_divide_value is None:
                    return True, "{0} = {1}, {2}".format(str(node.distinction.dimension_name),
                                                         str(c_index), path, mark)
                else:
                    return True, "{0} {3} {1}, {2}".format(str(node.distinction.dimension_name),
                                                           str(node.distinction.continuous_divide_value), path, mark)
        return False, ''


def find_idx_path(idx):
    cartpole = Problem_cartpole.CartPole()
    CUTreeAgent = Agent.CUTreeAgent(problem=cartpole, max_hist=opts.MAX_NODE_HIST,
                                    check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0,
                                    training_mode=opts.TRAINING_MODE)
    CUTreeAgent.read_Utree(game_number=165, save_path=CUTreeAgent.SAVE_PATH)
    utree = CUTreeAgent.utree
    # utree.print_tree_structure(CUTreeAgent.PRINT_TREE_PATH)

    flag, path = recursive_find_path(utree.root, idx)
    path_list = path.split(',')
    feature_value_dict = {}
    for path_section in path_list[:-2]:
        path_section = path_section.strip()
        path_section_list = path_section.split(' ')
        feature_name = path_section_list[0]
        value = float(path_section_list[2])

        if feature_value_dict.get(feature_name) is not None:
            feature_value_list = feature_value_dict.get(feature_name)

            if path_section_list[1] == '<':
                index = 1
                feature_value = feature_value_list[index] if feature_value_list[index] < value else value
                feature_value_list[index] = feature_value
            elif path_section_list[1] == '>':
                index = 0
                feature_value = feature_value_list[index] if value < feature_value_list[index] else value
                feature_value_list[index] = feature_value

            feature_value_dict.update({feature_name: feature_value_list})
        else:
            if path_section_list[1] == '<':
                feature_value_dict.update({feature_name: [-10000, value]})
            elif path_section_list[1] == '>':
                feature_value_dict.update({feature_name: [value, 10000]})
            else:
                feature_value_dict.update({feature_name: value})
    # CUTreeAgent.feature_importance()
    print feature_value_dict
    print 'path_length is {0}'.format(len(path_list[:-2]))
    print '{0}'.format(path_list[-2])
    print '{0}'.format(path_list[-1])

    cart_position_list = feature_value_dict.get('Cart_Position')
    if cart_position_list[0] == -10000:
        cart_position = cart_position_list[1] - 0.0000001
    elif cart_position_list[1] == 10000:
        cart_position = cart_position_list[0] + 0.0000001
    else:
        cart_position = sum(cart_position_list) / len(cart_position_list)

    cart_velocity_list = feature_value_dict.get('Cart_Velocity')
    if cart_velocity_list[0] == -10000:
        cart_velocity = cart_velocity_list[1] - 0.0000001
    elif cart_velocity_list[1] == 10000:
        cart_velocity = cart_velocity_list[0] + 0.0000001
    else:
        cart_velocity = sum(cart_velocity_list) / len(cart_velocity_list)

    pole_angle_list = feature_value_dict.get('Pole_Angle')
    if pole_angle_list[0] == -10000:
        pole_angle = pole_angle_list[1] - 0.0000001
    elif pole_angle_list[1] == 10000:
        pole_angle = pole_angle_list[0] + 0.0000001
    else:
        pole_angle = sum(pole_angle_list) / len(pole_angle_list)

    pole_velocity_at_tip_list = feature_value_dict.get('Pole_Velocity_At_Tip')
    if pole_velocity_at_tip_list[0] == -10000:
        pole_velocity_at_tip = pole_velocity_at_tip_list[1] - 0.0000001
    elif pole_velocity_at_tip_list[1] == 10000:
        pole_velocity_at_tip = pole_velocity_at_tip_list[0] + 0.0000001
    else:
        pole_velocity_at_tip = sum(pole_velocity_at_tip_list) / len(pole_velocity_at_tip_list)

    for action_choice in [0, 1]:
        instance = C_UTree.Instance(1000, [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip], action_choice, [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip], None, None)
        node = utree.getAbsInstanceLeaf(inst=instance)
        Q = node.qValues[action_choice]
        print 'idx {2}, action {0}: Q{1}'.format(action_choice, Q, node.idx)


if __name__ == "__main__":
    # idx = 1509
    # idx = 871
    idx = 537
    find_idx_path(idx)
