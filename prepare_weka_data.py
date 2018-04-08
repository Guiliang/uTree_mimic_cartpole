import csv
import copy
import dqn_cartpole
import numpy as np


def read_csv_game_record_dict(csv_dir):
    dict_all = []
    with open(csv_dir, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dict_all.append(row)
    return dict_all


def read_csv_game_record_list(csv_dir):
    dict_all = []
    with open(csv_dir, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            dict_all.append(row)
    return dict_all


def save_csv_record(csv_dir, data):
    with open(csv_dir, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in data:
            writer.writerow(val)


def generate_data():
    data_list = []
    for game_number in range(900, 1000):
        game_record = read_csv_game_record_dict(
            './save_all_transition/record_cartpole_transition_game{0}.csv'.format(int(game_number)))

        event_number = len(game_record)

        for index in range(0, event_number):
            transition = game_record[index]
            currentObs = transition.get('observation').split('$')
            nextObs = transition.get('newObservation').split('$')
            reward = float(transition.get('reward'))
            action = float(transition.get('action'))
            qValue = float(transition.get('qValue'))
            data_row = copy.copy(currentObs)
            data_row.append(action)
            data_row.append(qValue)
            data_list.append(data_row)
    save_csv_record('cartpole_testing_data.csv', data_list)


def generate_training_record_data():
    dict_all = read_csv_game_record_list('./record_training_observations.csv')

    learnStart = 128
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000
    deepQ = dqn_cartpole.DeepQ(4, 2, memorySize, discountFactor, learningRate, learnStart)
    deepQ.model = dqn_cartpole.load_model('cartpole-v0-save-r2.h5')

    for training_info_record in dict_all:
        observation = map(float, training_info_record[:-1])
        action = int(training_info_record[-1])

        qValues = deepQ.getQValues(np.asarray(observation))
        training_info_record.append(str(qValues[action]))

    save_csv_record('cartpole_dataset.csv', dict_all)



if __name__ == "__main__":
    generate_training_record_data()
