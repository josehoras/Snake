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



class SnakePart(pygame.sprite.Sprite):
    def __init__(self, grid, dir):
        super(SnakePart, self).__init__()
        self.body = pygame.Surface(sq_size)
        self.body.fill(white)
        self.grid = grid
        self.pos = (grid * sq_size).astype('float64')
        self.dir = dir
        self.rect = self.body.get_rect()
        self.rect.topleft = self.pos

    def update(self, speed):
        self.pos += speed * self.dir
        self.rect.topleft = self.pos



class SnakeFull(pygame.sprite.Group):
    def __init__(self):
        super(SnakeFull, self).__init__()
        self.body_grid = np.array(
            [[grid_x // 2 - 2, grid_y // 2],
             [grid_x // 2 - 1, grid_y // 2],
             [grid_x // 2, grid_y // 2]])
        self.direction = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        self.next_dir = self.direction[-1]
        self.length = 2
        self.speed = 0.1
        self.state = "alive"

        self.head = SnakePart(self.body_grid[-1], self.direction[-1])
        self.tail = SnakePart(self.body_grid[0], self.direction[0])
        for i in range(len(self.body_grid) - 1):
            self.add(SnakePart(self.body_grid[i+1], self.direction[i+1]))

        self.grid = self.update_grid()

    def update_grid(self):
        grid = [sp.grid for sp in self]
        grid.append(self.tail.grid)
        grid.append(self.head.grid)
        return grid

    def update_move(self, pressed_keys, number_pos):
        self.state = "alive"
        self.update_dir(pressed_keys)

        self.head.update(self.speed)
        if len(self) >= self.length:
            self.tail.update(self.speed)

        if (self.head.rect.topleft % sq_size == [0, 0]).all() and \
                (self.head.rect.topleft // sq_size != self.head.grid).any():
            self.head.grid = self.head.rect.topleft // sq_size
            self.head.dir = self.next_dir
            self.add(SnakePart(self.head.grid , self.head.dir))
            if (self.head.grid == number_pos).all():
                self.length += 2
                self.state = "just_ate"

        current_tail_grid = np.array([self.tail.pos // sq_size])
        if (self.tail.rect.topleft % sq_size == [0, 0]).all() and \
                (True in [(sp.grid == current_tail_grid).all() for sp in self]):
            to_del = [sp for sp in self if (sp.grid == current_tail_grid).all()]
            self.tail.dir = to_del[0].dir
            self.remove(to_del[0])

        if self.head.rect.top <= 0 or self.head.rect.bottom >= SCREEN_Y:
            self.state = "dead"
        if self.head.rect.left <= 0 or self.head.rect.right >= SCREEN_X:
            self.state = "dead"

        if len(pygame.sprite.spritecollide(self.head, self, False)) > 1:
            this_sp = pygame.sprite.spritecollide(self.head, self, False)
            self.state = "dead"

        self.update_grid()

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

#MAIN FUNCTION
SCREEN_X = 450
SCREEN_Y = 450

grid_x = 30
grid_y = 30

sq_size = np.array([SCREEN_X // (grid_x+1), SCREEN_Y // (grid_y+1)])

black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255


pygame.init()
screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))

snake = SnakeFull()

font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)

def generate_number(n, thing):
    number = n + 1
    c = [0]
    while c!=[]:
        number_grid = np.array([random.randint(0, grid_x), random.randint(0, grid_y)])
        c = [g for g in thing.grid if (number_grid == g).all()]
    number_txt = font.render(str(number), True, white)
    return number, number_grid, number_txt

number, number_grid, number_txt = generate_number(0, snake)
screen.blit(number_txt, number_grid * sq_size - [0, font_size/5])
pygame.display.update()
while not(check_continue_event()):
    pass


while not(check_quit_event()):
    snake.update_move(pygame.key.get_pressed(), number_grid)
    screen.fill(black)
    screen.blit(number_txt, number_grid * sq_size - [font_size/5 * (number>9), font_size/5])
    snake.plot()
    pygame.display.update()
    if snake.state == "just_ate":
        number, number_grid, number_txt = generate_number(number, snake)
    if snake.state == "dead":
        break

quit_msg = "Press Esc. to quit"
x = (screen.get_width() - font.size(quit_msg)[0]) / 2
y = (screen.get_height() - font.size(quit_msg)[1]) / 2
quit_surface = font.render(quit_msg, True, white)
screen.blit(quit_surface, (x, y))
pygame.display.update()
while not(check_quit_event()):
    pass

pygame.display.quit()