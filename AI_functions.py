import matplotlib.pyplot as plt
import torch
import numpy as np


def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def calculate_features(my_snake, number_grid, screen_grid):
    # print(my_snake.head.grid)
    # print(number_grid)
    head_pos = my_snake.head.grid.tolist()
    head_dir = my_snake.head.dir.tolist()
    # print(type(head_pos))
    # print(type(head_dir))
    # print(type(number_grid))
    distance_to_walls = [head_pos[0], head_pos[1], screen_grid[0] - head_pos[0], screen_grid[1] - head_pos[1]]
    distance_to_number = [number_grid[0] - head_pos[0], number_grid[1] - head_pos[1]]
    # features = [distance_to_walls + distance_to_number + head_dir]
    features = [distance_to_walls + distance_to_number + head_pos + head_dir + number_grid.tolist()]
    features = torch.tensor(features, dtype=torch.float)
    return features


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def normalize_rewards(r, gamma=0.99):
    norm_r = discount_rewards(r.cpu().numpy(), gamma)
    norm_r -= np.mean(norm_r)
    if np.std(norm_r) != 0: norm_r /= np.std(norm_r)
    return norm_r

def update_performance(performance, number, level_moves):
    performance['score'].append(number-1)
    smooth_score = np.sum(performance['score'][-100:]) / len(performance['score'][-100:])
    performance['smooth_score'].append(smooth_score)
    performance['moves'].append(level_moves)
    smooth_move = np.sum(performance['moves'][-100:])/len(performance['moves'][-100:])
    performance['smooth_moves'].append(smooth_move)
    return performance


def plot_screens(screens):
    rows = 3
    cols = 3
    n = rows * cols
    fig = plt.figure(figsize=(cols*2.25, rows*2.25), dpi=100)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)  # sets the number of feature maps to show on each row and column
        j = i * 1
        plt.title(str(j), fontsize=10)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(screens[i], cmap='Greys')
    plt.tight_layout()
    fig.savefig('screens_plot.png')
    plt.close("all")
    # plt.show()


def plot_performance(performance, file_base_name):
    fig = plt.figure(figsize=(7, 7), dpi=100)

    plt.subplot(2, 1, 1)
    plt.title("total moves", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot(performance['moves'])
    plt.plot(performance['smooth_moves'])
    plt.subplot(2, 1, 2)
    plt.title("score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot(performance['score'])#, color='g')
    plt.plot(performance['smooth_score'])
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    # # plt.plot(performance['score'])
    # ax1.plot(performance['moves'])
    # ax1.plot(performance['smooth_moves'])
    # ax2.plot(performance['score'], color='g')
    # ax2.set_ylim([0, 10])
    plt.tight_layout()
    fig.savefig(file_base_name + '.png')
    plt.close("all")
    # plt.show()

def plot_value(value, x, y):
    fig = plt.figure(figsize=(7, 7), dpi=100)
    value_plane = value[:, :, x, y]
    print("Plane shape: ", value_plane.shape)
    print(value_plane[0, 9])
    value_plane_dir = np.mean(value_plane, axis=3)

    print("Plane shape: ", value_plane.shape)
    print(value_plane[0, 9])
    d=0
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.title("action: " + str(i), fontsize=14)
        # plt.xticks([])
        # plt.yticks([])

        plt.imshow(value_plane[:,:,0,i], #extent=(x.min(), x.max(), y.max(), y.min()),
           interpolation='nearest', cmap='gist_heat')
        plt.colorbar()
        d +=1
    plt.tight_layout()
    fig.savefig('td/value.png')
    plt.close("all")