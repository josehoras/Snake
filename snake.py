import random
import numpy as np
import pygame
from pygame.locals import *

def check_quit_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            return True
        elif event.type == QUIT:
            return True
    return False


def check_continue_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_SPACE:
            return True
    return False


def check_pause_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_SPACE:
            return True
    return False


class SnakePart(pygame.sprite.Sprite):
    def __init__(self, grid, dir, style, sq_size):
        super(SnakePart, self).__init__()
        self.dir = dir
        self.body = pygame.Surface(sq_size)
        self.rect = self.body.get_rect()
        if style == "square":
            self.body.fill(white)
            pygame.draw.rect(self.body, black, self.rect, 1)
        if style == "round":
            pygame.draw.circle(self.body, (0, 255, 0), sq_size//2 , min(sq_size)//2)
        # pygame.draw.rect(self.body, blue, self.rect, 0)
        # pygame.draw.line(self.body, white, (0,0), abs(sq_size * self.dir))
        # pygame.draw.line(self.body, white, (sq_size - (1,1)) * abs(self.dir[::-1]), sq_size - (1,1) )

        self.grid = grid
        self.pos = (grid * sq_size).astype('float64')
        self.rect.topleft = self.pos

    def update(self, speed):
        self.pos += speed * self.dir
        self.rect.topleft = self.pos


class Snake(pygame.sprite.Group):
    def __init__(self, style, screen_size, grid_size):
        super(Snake, self).__init__()
        self.grid = grid_size//2 - [[2,0], [1,0], [0,0]]
        self.direction = np.array([[1, 0], [1, 0], [1, 0]])

        self.screen_size = screen_size
        self.sq_size = screen_size // grid_size
        self.style = style
        self.next_dir = self.direction[-1]
        self.length = 3
        self.length_increase = 10
        self.speed = 0.05
        self.state = "alive"

        self.tail = SnakePart(self.grid[0], self.direction[0], self.style, self.sq_size)
        for i in range(1, len(self.grid)):
            self.add(SnakePart(self.grid[i], self.direction[i], self.style, self.sq_size))
        self.head = SnakePart(self.grid[-1], self.direction[-1], self.style, self.sq_size)

    def update_move(self, pressed_keys, number_pos):
        self.state = "alive"
        # Check user input
        self.update_dir(pressed_keys)
        # Move the head and, if the length is ok, the tail
        self.head.update(self.speed)
        if len(self) >= self.length:
            self.tail.update(self.speed)
        # Check if head has reached a new grid square and add a new part to body
        if (self.head.rect.topleft % self.sq_size == [0, 0]).all() and \
                (self.head.rect.topleft // self.sq_size != self.head.grid).any():
            self.head.grid = self.head.rect.topleft // self.sq_size
            self.head.dir = self.next_dir
            self.add(SnakePart(self.head.grid, self.head.dir, self.style, self.sq_size))
            self.grid = np.append(self.grid, [self.head.grid], axis=0)
            if (self.head.grid == number_pos).all():
                self.length += self.length_increase
                self.state = "just_ate"
        # Check if tail has reached a new grid square and remove that part of the body
        if (self.tail.rect.topleft % self.sq_size == [0, 0]).all() and \
                (self.tail.rect.topleft // self.sq_size != self.tail.grid).any():
            to_del = [sp for sp in self if (sp.grid == self.tail.rect.topleft // self.sq_size).all()]
            self.grid = self.grid[[(g != self.tail.grid).any() for g in self.grid]]
            # print([np.array2string(i, separator=',') for i in self.grid])
            self.tail.dir = to_del[0].dir
            self.tail.grid = to_del[0].grid
            self.remove(to_del[0])
        # Check dying conditions
        if self.head.rect.top < 0 or self.head.rect.bottom > self.screen_size[1]:
            self.state = "dead"
        if self.head.rect.left < 0 or self.head.rect.right > self.screen_size[0]:
            self.state = "dead"
        if len(pygame.sprite.spritecollide(self.head, self, False)) > 1:
            self.state = "dead"

    def update_dir(self, pressed_keys):
        if pressed_keys[K_LEFT] and self.head.dir[0]!=1:
            self.next_dir = np.array([-1, 0])
        if pressed_keys[K_RIGHT] and self.head.dir[0]!=-1:
            self.next_dir = np.array([1, 0])
        if pressed_keys[K_UP] and self.head.dir[1]!=1:
            self.next_dir = np.array([0, -1])
        if pressed_keys[K_DOWN] and self.head.dir[1]!=-1:
            self.next_dir = np.array([0, 1])

    def plot(self, screen):
        screen.blit(self.head.body, self.head.rect)
        screen.blit(self.tail.body, self.tail.pos)
        for sp in self:
            screen.blit(sp.body, sp.pos)


# Functions to generte numbers and plot messages
def generate_number(n, thing):
    n += 1
    c = [0]
    n_grid = []
    while c:
        n_grid = np.array([random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1)])
        c = [g for g in thing.grid if (n_grid == g).all()]
    n_txt = font.render(str(n), True, white)
    return n, n_grid, n_txt


def plot_msg(msg):
    x = (screen.get_width() - font.size(msg)[0]) / 2
    y = (screen.get_height() - font.size(msg)[1]) / 2
    msg_surface = font.render(msg, True, (250, 0, 0))
    screen.blit(msg_surface, (x, y))


#MAIN FUNCTION
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
sq_size = screen_size // grid_size

# Start pygame screen
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake')
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)

# Create snake and first number
snake = Snake("square", screen_size, grid_size)  # style "square" or "round"
number, number_grid, number_txt = generate_number(0, snake)
screen.blit(number_txt, number_grid * sq_size - [0, font_size/5])
snake.plot(screen)
plot_msg("Press Space to start")
pygame.display.update()
while not(check_continue_event()):
    pass

# Main loop
while not(check_quit_event()):
    snake.update_move(pygame.key.get_pressed(), number_grid)
    screen.fill(black)
    screen.blit(number_txt, number_grid * sq_size - [font_size / 5 * (number > 9), font_size / 5])
    snake.plot(screen)
    pygame.display.update()
    if snake.state == "just_ate":
        number, number_grid, number_txt = generate_number(number, snake)
    if snake.state == "dead":
        break

plot_msg("Press Esc. to quit")
pygame.display.update()
while not(check_quit_event()):
    pass

pygame.display.quit()