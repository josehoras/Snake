from snake_objects import Snake
from snake_functions import *
from AI_models import *
from AI_functions import *
import os.path
import pickle
import torch



# MAIN FUNCTION
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
sq_size = screen_size // grid_size
render = True

# Define model


# Maybe load previous model
gamma = 0.9
reward_p = 1
file_dir = "td/"
file_base = "modelo-small"
file_post = "_5"
file_ext = ".td"
file_name = file_dir + file_base + file_post + file_ext

output_dir = "./"


if os.path.isfile(file_name) == True:
    value, performance, it = pickle.load(open(file_name, 'rb'), encoding='latin1')
else:
    value = np.full((grid_size[0], grid_size[1], grid_size[0], grid_size[1], 4, 5), 0.2)
    performance = {'score':[], 'smooth_score':[], 'moves':[], 'smooth_moves':[]}
    it = 0

# Start pygame screen
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake')
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)
delay = 300
record_dir = "/video/"

set_n = 4
alpha = 10
gamma = 1
print(value.shape)
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
d_name = ['left', 'right', 'up', 'down']
a_name = ['left', 'right', 'up', 'down', 'no']
# Main loop
for i in range(it + 1, it + 20001):
    # Create snake and first number
    snake = Snake("square", screen_size, grid_size, speed=5)  # style "square" or "round"
    number, number_grid, number_txt = generate_number(0, snake.grid, grid_size,
                                                      font, white, set=[4, 4])
    update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)

    level_moves = 0
    number_moves = 0
    while not(check_quit_event()):
        level_moves += 1
        number_moves += 1

        # features = calculate_features(snake, number_grid, grid_size).to(device)

        old_dir = [i for i in range(4) if (directions[i]==snake.head.dir).all()][0]

        old_state = (snake.head.grid[0], snake.head.grid[1], number_grid[0], number_grid[1], old_dir)
        probs_td = np.exp(value[old_state])/sum(np.exp(value[old_state]))

        # print("probs: ", probs_td)
        # Take action depending on prob of each option.
        action = np.random.choice(range(5), p=probs_td)  # policy
        # snake.update_move(action, number_grid, mode='AI')
        snake.update_move(pygame.key.get_pressed(), number_grid)
        new_dir = [i for i in range(4) if (directions[i]==snake.head.dir).all()][0]
        new_state = (snake.head.grid[0], snake.head.grid[1], number_grid[0], number_grid[1], new_dir)

        reward = torch.tensor([0], dtype=torch.float)
        if snake.state == "just_ate":
            number_moves = 0
            reward += 1
            number, number_grid, number_txt = generate_number(number, snake.grid, grid_size,
                                                              font, white, set=[4, 4])
        if snake.state == "dead":
            reward += -1
            print("At ", snake.head.grid[0], snake.head.grid[1], "going ", d_name[old_dir])
            print("value: ", value[old_state], "on ", old_state)
            print("probs: ", probs_td)
            print("action taken: %s (%i)" % (a_name[action], action))
            print("new dir: ", d_name[new_dir], "on ", new_state)

            value[old_state][action] += alpha * reward
            print("new value: ", value[old_state])
            break
        if number_moves == 1000:
            reward += 0
            # update this value
            # value[state][action] += alpha * (reward + gamma*np.max(value[new_state]) - value[state][action])
            break
        if number > 50:
            reward += 1
            # update this value
            value[old_state][action] += alpha * (reward + gamma*np.max(value[new_state]) - value[old_state][action])
            break


        # print(np.mean(value[new_state]) - value[state][action])
        # if np.mean(value[new_state]) - value[state][action] > 0.01:
        #     print(value[state])
        # inc = reward + gamma*np.max(value[new_state]) - value[state][action]
        # if inc > 0.01:
        #     print("At ", snake.head.grid[0], snake.head.grid[1], "dir: ", dir)
        #     print(value[state])
        # value[state][action] += inc
        # if inc > 0.01: print(value[state])
        # if value[state][action] < 0: value[state][action]=0
        # if np.mean(value[new_state]) - value[state][action] > 0.01:
        #     print(value[state])
        #     pygame.time.delay(1000)
        update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)
        delay = check_delay(delay, pygame.key.get_pressed())

        pygame.time.delay(delay)


    performance = update_performance(performance, number, level_moves)
    print()
    print(i, " - score: %i (%.1f), moves: %i (%.0f)" %
          (performance['score'][-1], performance['smooth_score'][-1],
           performance['moves'][-1], performance['smooth_moves'][-1]))
    print()
    if i % 50 == 0:
        plot_performance(performance, file_dir + file_base + file_post)
        plot_value(value, 4, 4)
        pickle.dump([value, performance, i], open(file_dir + file_base + file_post + file_ext, 'wb'))


# plot_msg("Press Esc. to quit", screen, font)
# pygame.display.update()
# while not(check_quit_event()):
#     pass

pygame.display.quit()
