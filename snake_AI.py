from snake_objects import Snake
from snake_functions import *
from AI_models import *
from AI_functions import *
import os.path
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchsummary import summary

MODELS = {"features_2_layers": features_2_layers,
          "features_linear_5": features_linear_5_net,
          "features_linear_big": features_linear_big_net}
OPTIMIZERS = {"Adam": Adam}

# Function with global variables for code economy
def update_series(features, action, reward):
    # input_series.append(input.squeeze(0))
    features_series.append(features.squeeze(0))
    action_series.append(action)
    reward_series.append(reward)


# MAIN FUNCTION
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
sq_size = screen_size // grid_size
render = False

# Define model
device = torch.device("cpu")
learning_rate = 8e-4
l2_reg = 5e-5
dropout = 0
model_name = "features_2_layers"
model_class = MODELS[model_name]
model = model_class().to(device)
print(model)
optimizer_name = "Adam"
optimizer_class = OPTIMIZERS[optimizer_name]
optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
loss_fn = nn.CrossEntropyLoss(reduction='none')
softmax_fn = torch.nn.Softmax(dim=1)
T = 0.1

# Maybe load previous model
gamma = 0.9
reward_p = 1
file_dir = "reward/"
file_base = "2_layers"
file_post = "_8-4"
file_ext = ".snake"
file_name = file_dir + file_base + file_post + file_ext

output_dir = "./"
it = 0
performance = {'score':[], 'smooth_score':[], 'moves':[], 'smooth_moves':[]}
if os.path.isfile(file_name) == True:
    checkpoint = torch.load(output_dir + file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = learning_rate
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint['optimizer_state_dict']['param_groups'])
    print(checkpoint.keys())
    # print(summary(model.cuda(), (1, 8)))
    # model.cpu()
    performance = checkpoint['performance']
    it = checkpoint['iterations']


# Start pygame screen
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake')
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)
delay = 0
record_dir = "/video/"


# Main loop
for i in range(it + 1, it + 20001):
    # Create snake and first number
    snake = Snake("square", screen_size, grid_size, speed=5)  # style "square" or "round"
    number, number_grid, number_txt = generate_number(0, snake.grid, grid_size, font, white)#, set=[20, 8])

    update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)

    # input_series = []
    features_series, action_series, reward_series = [], [], []
    level_moves = 0
    number_moves = 0
    adventure = ''
    n_ad = 0
    # frame_old = np.zeros(screen_size)
    model.eval()
    while not(check_quit_event()):
        level_moves += 1
        number_moves += 1
        # frame_new = pygame.surfarray.array2d(screen)
        # input = torch.tensor((frame_old-frame_new)/2 + frame_new, dtype=torch.float).to(device)
        # input = torch.unsqueeze(input, 0)
        # frame_old = frame_new
        features = calculate_features(snake, number_grid, grid_size).to(device)

        scores = model(features)
        # scores = scores / T
        # scores = model(input)
        probs = softmax_fn(scores/T)
        # Take action depending on prob of each option. If it is too sure, sometimes try a random action
        if torch.max(probs) > 1.9 and random.randint(0, 40)==1:
            action = torch.randint(0, 4, (1,))      # fully random action
            n_ad += 1
            adventure = '(' + str(n_ad) + '*)'
        else:
            action = torch.multinomial(probs, 1).reshape(1)
        # print(action)
        action_taken = snake.update_move(action, number_grid, mode='AI')
        if action_taken:
            reward = torch.tensor([0], dtype=torch.float)
            if snake.state == "just_ate":
                number_moves = 0
                reward += reward_p
                number, number_grid, number_txt = generate_number(number, snake.grid, grid_size, font, white)
            if snake.state == "dead":
                reward += -1
                update_series(features, action, reward)
                break
            if number > 50:
                reward += 1
                update_series(features, action, reward)
                break
            update_series(features, action, reward)
        if number_moves == 500:
            reward = 0
            update_series(features, action, reward)
            break
        update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)
        delay = check_delay(delay, pygame.key.get_pressed())

        pygame.time.delay(delay)

    # input_series = torch.stack(input_series).to(device)
    features_series = torch.stack(features_series).to(device)
    action_series = torch.as_tensor(action_series).to(device)
    reward_series = torch.as_tensor(reward_series).to(device)

    norm_rewards = torch.tensor(normalize_rewards(reward_series, gamma), dtype=torch.float).to(device)
    model.train()
    scores_series = model(features_series)
    losses = loss_fn(scores_series, action_series)
    loss = torch.mean(losses * norm_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    performance = update_performance(performance, number, level_moves)

    print(i, " - loss: %.1e, score: %i (%.1f), moves: %i (%.0f) %s" %
          (loss.data.item(), performance['score'][-1], performance['smooth_score'][-1],
           performance['moves'][-1], performance['smooth_moves'][-1], adventure))
    if i % 50 == 0:
        plot_performance(performance, file_dir + file_base + file_post)
        torch.save({'model_name': model_name,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer_name': optimizer_name,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': learning_rate,
                    'gamma': gamma,
                    'performance': performance,
                    'iterations': i},
                   output_dir + file_name)


# plot_msg("Press Esc. to quit", screen, font)
# pygame.display.update()
# while not(check_quit_event()):
#     pass

pygame.display.quit()
