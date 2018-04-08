import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
sns.set(font_scale=2)


def read_record_csv(csv_dir):
    read_counter = 0
    correlations_array = None
    mse_array = None
    mae_array = None
    with open(csv_dir, 'rb') as file:
        rows = file.readlines()
        for row in rows:
            if row == '\n':
                continue
            else:
                # print row
                row = row.replace('\n', '')
                read_counter += 1
                if read_counter % 3 == 1:
                    mae_split = row.split(',')
                    mae_list = map(float, mae_split[2:])
                    if mae_array is None:
                        mae_array = np.asarray([np.asarray(mae_list)])
                    else:
                        mae_array = np.concatenate((mae_array, [np.asarray(mae_list)]), axis=0)
                elif read_counter % 3 == 2:
                    mse_split = row.split(',')
                    mse_list = map(float, mse_split[2:])
                    if mse_array is None:
                        mse_array = np.asarray([np.asarray(mse_list)])
                    else:
                        mse_array = np.concatenate((mse_array, [np.asarray(mse_list)]), axis=0)
                elif read_counter % 3 == 0:
                    correl_split = row.split(',')
                    correl_list = map(float, correl_split[2:])
                    if correlations_array is None:
                        correlations_array = np.asarray([np.asarray(correl_list)])
                    else:
                        correlations_array = np.concatenate((correlations_array, [np.asarray(correl_list)]), axis=0)
    return mae_array, mse_array, correlations_array


def draw_correl(correl_array):
    x = [400 * x for x in range(0, len(correl_array[:, 0]))]

    y = correl_array.transpose()

    for index in range(0, len(correl_array)):
        correl_array[index] = smooth(correl_array[index], 1)

    plot_tree_shadow_result(x, y, 'Correlation')


def draw_mae(array_mae):
    x = [400 * x for x in range(0, len(array_mae[:, 0]))]

    y = array_mae.transpose()

    plot_tree_shadow_result(x, y, 'MAE')


def draw_rmse(array_mse):
    x = [400 * x for x in range(0, len(array_mse[:, 0]))]

    y = array_mse.transpose()

    y = [yy ** 0.5 for yy in y]

    plot_tree_shadow_result(x, y, 'RMSE')


def plot_tree_shadow_result(x_plot_array, y_plot_array, name):
    x_plot_array = [float(number) / 1000 for number in x_plot_array]
    plt.figure(figsize=(10, 6.5))
    ax = sns.tsplot(y_plot_array, x_plot_array, condition=name, legend=True)
    ax.ticklabel_format(axis='x', style='sci')
    plt.xlabel('Transition Numbers (by thousands)')
    # plt.ylabel(name)
    # plt.legend(loc='best')
    plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":
    mae_array, mse_array, correlations_array = read_record_csv(csv_dir='./result/flappy_bird_mimic.csv')
    draw_correl(correlations_array)
    draw_mae(mae_array)
    draw_rmse(mse_array)
