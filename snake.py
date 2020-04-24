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
    def __init__(self, grid, dir, style):
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


class SnakeFull(pygame.sprite.Group):
    def __init__(self, style):
        super(SnakeFull, self).__init__()
        self.body_grid = np.array(
            [[grid_x // 2 - 2, grid_y // 2],
             [grid_x // 2 - 1, grid_y // 2],
             [grid_x // 2, grid_y // 2]])
        self.direction = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])

        self.style = style
        self.next_dir = self.direction[-1]
        self.length = 3
        self.length_increase = 10
        self.speed = 0.06
        self.state = "alive"

        self.head = SnakePart(self.body_grid[-1], self.direction[-1], self.style)
        self.tail = SnakePart(self.body_grid[0], self.direction[0], self.style)
        for i in range(len(self.body_grid) - 1):
            self.add(SnakePart(self.body_grid[i+1], self.direction[i+1], self.style))

        self.grid = self.update_grid()

    def update_grid(self):
        grid = [sp.grid for sp in self]
        grid.append(self.tail.grid)
        grid.append(self.head.grid)
        return grid

    def update_move(self, pressed_keys, number_pos):
        self.state = "alive"
        # Check user input
        self.update_dir(pressed_keys)
        # Move the head and, if the length is ok, the tail
        self.head.update(self.speed)
        if len(self) >= self.length:
            self.tail.update(self.speed)
        # Check if head has reached a new grid square and add a new part to body
        if (self.head.rect.topleft % sq_size == [0, 0]).all() and \
                (self.head.rect.topleft // sq_size != self.head.grid).any():
            self.head.grid = self.head.rect.topleft // sq_size
            self.head.dir = self.next_dir
            self.add(SnakePart(self.head.grid , self.head.dir, self.style))
            self.grid = self.update_grid()
            if (self.head.grid == number_pos).all():
                self.length += self.length_increase
                self.state = "just_ate"
        # Check if tail has reached a new grid square and remove that part of the body
        if (self.tail.rect.topleft % sq_size == [0, 0]).all() and \
                (self.tail.rect.topleft // sq_size != self.tail.grid).any():
            to_del = [sp for sp in self if (sp.grid == self.tail.rect.topleft // sq_size).all()]
            self.tail.dir = to_del[0].dir
            self.tail.grid = to_del[0].grid
            self.remove(to_del[0])
            self.grid = self.update_grid()
        # Check dying conditions
        if self.head.rect.top < 0 or self.head.rect.bottom >= SCREEN_Y:
            self.state = "dead"
        if self.head.rect.left < 0 or self.head.rect.right >= SCREEN_X:
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

    def plot(self):
        screen.blit(self.head.body, self.head.rect)
        screen.blit(self.tail.body, self.tail.pos)
        for sp in self:
            screen.blit(sp.body, sp.pos)

# Functions to generte numbers and plot messages
def generate_number(n, thing):
    number = n + 1
    c = [0]
    while c!=[]:
        number_grid = np.array([random.randint(0, grid_x), random.randint(0, grid_y)])
        c = [g for g in thing.grid if (number_grid == g).all()]
    number_txt = font.render(str(number), True, white)
    return number, number_grid, number_txt


def plot_msg(msg):
    x = (screen.get_width() - font.size(msg)[0]) / 2
    y = (screen.get_height() - font.size(msg)[1]) / 2
    msg_surface = font.render(msg, True, (250, 0, 0))
    screen.blit(msg_surface, (x, y))


#MAIN FUNCTION
SCREEN_X = 400
SCREEN_Y = 400
grid_x = 20
grid_y = 20
sq_size = np.array([SCREEN_X // (grid_x+1), SCREEN_Y // (grid_y+1)])
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
# Start pygame screen
pygame.init()
screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)

# Create snake and first number
snake = SnakeFull("square")  # style "square" or "round"
number, number_grid, number_txt = generate_number(0, snake)
screen.blit(number_txt, number_grid * sq_size - [0, font_size/5])
snake.plot()
plot_msg("Press Space to start")
pygame.display.update()
while not(check_continue_event()):
    pass

# Main loop
while not(check_quit_event()):
    snake.update_move(pygame.key.get_pressed(), number_grid)
    screen.fill(black)
    screen.blit(number_txt, number_grid * sq_size - [font_size / 5 * (number > 9), font_size / 5])
    snake.plot()
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