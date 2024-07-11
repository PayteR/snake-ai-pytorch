import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from lib.config import IS_PLOT_RENDERED, IS_GAME_RENDERED, TORCH_DEVICE, MAX_MEMORY, BATCH_SIZE, LR, EPSILON


class Agent:

    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]

        # Clok-wise directions and angles
        cw_dirs = [
          Direction.RIGHT == game.direction, 
          Direction.DOWN == game.direction,
          Direction.LEFT == game.direction,
          Direction.UP == game.direction
          ]
        cw_angs = np.array([0, np.pi/2, np.pi, -np.pi/2])

        # Position - in front: 0, on right: 1, on left: -1; BLOCK_SIZE = 20
        getPoint = lambda pos: Point(
          head.x + 20*np.cos(cw_angs[(cw_dirs.index(True)+pos) % 4]),
          head.y + 20*np.sin(cw_angs[(cw_dirs.index(True)+pos) % 4]))

        headingPoint = getPoint(0)
        state = [
          # Danger
          game.is_collision_predict(headingPoint, headingPoint, head),
          game.is_collision_predict(getPoint(1), headingPoint, head),
          game.is_collision_predict(getPoint(-1), headingPoint, head),

          # Move direction
          cw_dirs[2],
          cw_dirs[0],
          cw_dirs[3],
          cw_dirs[1],

          # Food location
          game.food.x < head.x,
          game.food.x > head.x,
          game.food.y < head.y,
          game.food.y > head.y
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        eps = EPSILON - self.n_games
        final_move = [0,0,0]
        if eps > 0 and random.randint(0, EPSILON * 2) < eps:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(TORCH_DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).to(TORCH_DEVICE).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if not done:
            continue
        
        # train long memory, plot result
        game.reset()
        agent.n_games += 1
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()
        elif agent.n_games % 100 == 0:
            agent.model.save()

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        if IS_PLOT_RENDERED:
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()