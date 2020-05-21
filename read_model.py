#! /usr/bin/env python

from AI_models import *
import torch
from torch.optim import Adam, SGD
from snake_functions import *
import pygame
from pygame.locals import *
import numpy as np
import argparse
#torch.nn.Module.dump_patches = True

MODELS = {"features_linear":features_linear_net,
          "features_linear_5":features_linear_5_net,
          "features_linear_big":features_linear_big_net}
OPTIMIZERS = {"Adam": Adam}

white = 255, 255, 255


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name",
                        help="The file name of the model")
    args = parser.parse_args()
    return args

def print_msg(screen, font, text, x, y):
    txt_surf = font.render(text, True, white)
    screen.blit(txt_surf, (x, y))

def load_model(checkpoint):
    # model_name = checkpoint['model_name']
    # model_class = MODELS[model_name]
    # model = model_class().to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    #
    # learning_rate = checkpoint['lr']
    # optimizer_name = checkpoint['optimizer_name']
    # optimizer_class = OPTIMIZERS[optimizer_name]
    # optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    #
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # performance = checkpoint['performance']
    pass


def main(args):
    # Start pygame screen
    screen_size = np.array([400, 400])
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption('Snake')
    # Define fonts for screen messages
    font_size = 24
    font = pygame.font.SysFont("ubuntumono", font_size)

    # Open model
    file_name = args.file_name
    device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(file_name)

    # write model_card
    model_card = {}
    model_card['model_name'] = checkpoint['model_name']
    model_card['optimizer'] = checkpoint['optimizer_name']
    model_card['learning_rate'] = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    model_card['l2_reg'] = checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay']
    model_card['iterations'] = checkpoint['iterations']
    model_card['avg_score'] = checkpoint['performance']['smooth_score'][-1]
    # model_card['gamma'] = checkpoint['gamma']

    i = 0
    for key, value in model_card.items():
        y = 10 + 30 * i
        text = key + ": {0}".format(model_card[key])
        print_msg(screen, font, text, 10, y)
        i += 1
        if i in [1, 5]: i += 1


    pygame.display.update()
    while not(check_quit_event()):
        pass

if __name__ == "__main__":
    main(get_args())