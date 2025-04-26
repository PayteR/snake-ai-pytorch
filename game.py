from lib.dotenvload import os
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from lib.config import IS_GAME_RENDERED, BLOCK_SIZE, SPEED

if(IS_GAME_RENDERED):
    pygame.init()
    font = pygame.font.Font('arial.ttf', 25)
    #font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0,200,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.grid_h = self.h // BLOCK_SIZE
        self.grid_w = self.w // BLOCK_SIZE
        # init display
        if(IS_GAME_RENDERED):
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        # Place head in the center, ensuring it's aligned with the grid
        center_x = (self.grid_w // 2) * BLOCK_SIZE
        center_y = (self.grid_h // 2) * BLOCK_SIZE
        self.head = Point(center_x, center_y)

        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        # Place food randomly on the grid
        x = random.randint(0, self.grid_w - 1) * BLOCK_SIZE
        y = random.randint(0, self.grid_h - 1) * BLOCK_SIZE
        self.food = Point(x, y)
        # Ensure food is not placed inside the snake
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input (only relevant if rendering/debugging)
        if IS_GAME_RENDERED:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        # Increase penalty for collision, consider adding penalty for long survival without eating
        # Add check for frame iteration limit to prevent infinite loops
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake): # Increased timeout factor
            game_over = True
            reward = -10 # Keep penalty for dying
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 # Reward for eating food
            self._place_food()
        else:
            self.snake.pop()
            # Optional: Small penalty for each step to encourage faster eating
            # reward = -0.01

        # 5. update ui and clock
        if IS_GAME_RENDERED:
            self._update_ui()
            self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def get_state_image(self):
        """
        Returns a 2D numpy array representing the game state.
        Pixel values: 0: empty, 0.25: food, 0.5: snake tail, 0.75: snake body, 1.0: snake head
        """
        # Initialize grid with zeros (empty)
        image = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Place food (ensure coordinates are within grid bounds)
        food_x_grid, food_y_grid = int(self.food.x // BLOCK_SIZE), int(self.food.y // BLOCK_SIZE)
        if 0 <= food_y_grid < self.grid_h and 0 <= food_x_grid < self.grid_w:
             image[food_y_grid, food_x_grid] = 0.25

        # Place snake body and tail
        for i, pt in enumerate(self.snake):
            body_x_grid, body_y_grid = int(pt.x // BLOCK_SIZE), int(pt.y // BLOCK_SIZE)
            if 0 <= body_y_grid < self.grid_h and 0 <= body_x_grid < self.grid_w:
                if i == len(self.snake) - 1: # Tail
                    image[body_y_grid, body_x_grid] = 0.5
                elif i > 0: # Body segment (excluding head)
                    image[body_y_grid, body_x_grid] = 0.75

        # Place snake head (ensure coordinates are within grid bounds)
        head_x_grid, head_y_grid = int(self.head.x // BLOCK_SIZE), int(self.head.y // BLOCK_SIZE)
        if 0 <= head_y_grid < self.grid_h and 0 <= head_x_grid < self.grid_w:
            image[head_y_grid, head_x_grid] = 1.0

        # Add channel dimension (C, H, W) - PyTorch expects channels first
        image = np.expand_dims(image, axis=0)
        return image


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            if pt == self.head:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)