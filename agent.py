import torch
import random
import numpy as np
from collections import deque
# Import game constants and classes
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
# Import the new CNN model and trainer
from model import Conv_QNet, QTrainer, MODEL_FILENAME # Use new model filename
from helper import plot
from lib.config import IS_PLOT_RENDERED, IS_GAME_RENDERED, TORCH_DEVICE, MAX_MEMORY, BATCH_SIZE, LR, EPSILON
import os

# Constants for the CNN Agent
# INPUT_CHANNELS = 1 # Number of channels in the input image (1 for grayscale)
# FRAME_STACK = 4 # Number of frames to stack (optional, start with 1)
# INPUT_CHANNELS = FRAME_STACK if FRAME_STACK > 1 else 1
INPUT_CHANNELS = 1 # Start simple with single frame
OUTPUT_SIZE = 3 # Number of actions: straight, right, left

# Hyperparameters (consider tuning these)
GAMMA = 0.9 # Discount rate (keep relatively high)
# Epsilon parameters for decay
# EPSILON_START = 1.0 # Start with high exploration
# EPSILON_END = 0.05 # End with low exploration
# EPSILON_DECAY_STEPS = 200000 # Number of steps to decay epsilon over (adjust based on training time)
INITIAL_EPSILON_GAMES = EPSILON # Use the value from config as the number of games for initial linear decay
MIN_EPSILON_GAMES_FACTOR = 0.1 # Minimum epsilon as a factor of initial games (e.g., 80 * 0.1 = 8)


class Agent:

    def __init__(self, game_instance): # Pass game instance to get dimensions
        self.n_games = 0
        self.epsilon_threshold = INITIAL_EPSILON_GAMES # Initial exploration games
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        # Get game dimensions for the CNN input size
        self.input_h = game_instance.grid_h
        self.input_w = game_instance.grid_w

        # Create main model and target model using Conv_QNet
        self.model = Conv_QNet(INPUT_CHANNELS, self.input_h, self.input_w, OUTPUT_SIZE)
        self.target_model = Conv_QNet(INPUT_CHANNELS, self.input_h, self.input_w, OUTPUT_SIZE)

        # Load existing model weights if available
        self.model.load() # Load main model weights using MODEL_FILENAME
        self.target_model.load_state_dict(self.model.state_dict()) # Sync target model initially
        self.target_model.eval() # Set target model to evaluation mode

        # Initialize trainer with both models and learning rate from config
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        # TODO: Consider saving/loading optimizer state and train_step_counter for resuming training properly


    def get_state(self, game):
        # Get the processed image state from the game
        # The image is already (1, H, W) numpy array
        state_image = game.get_state_image()
        # Optional: Implement frame stacking here if desired
        # If using frame stacking, manage the stack in the agent or game
        return state_image


    def remember(self, state, action, reward, next_state, done):
        # State and next_state are now image tensors (or numpy arrays)
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        elif len(self.memory) > 0: # Train even if less than BATCH_SIZE if memory is filling up
             mini_sample = self.memory
        else:
            return # Not enough memory to train

        # Check if mini_sample is not empty before proceeding
        if not mini_sample:
            return

        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Convert lists of states/next_states (which are numpy arrays) into a single batch tensor
        states_batch = np.array(states)
        next_states_batch = np.array(next_states)

        # Pass data to the trainer's train_step method
        self.trainer.train_step(states_batch, actions, rewards, next_states_batch, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        # State and next_state are single image numpy arrays (1, H, W)
        # Trainer's train_step handles adding the batch dimension if needed
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-greedy strategy with linear decay over games
        # Calculate current epsilon based on number of games played
        # Decay linearly from 1.0 (implicitly, via random chance) down to a minimum
        # The threshold determines if we explore or exploit
        current_epsilon_threshold = self.epsilon_threshold - self.n_games
        # Ensure threshold doesn't go below a minimum exploration level
        min_threshold = INITIAL_EPSILON_GAMES * MIN_EPSILON_GAMES_FACTOR
        final_threshold = max(min_threshold, current_epsilon_threshold)

        final_move = [0, 0, 0] # [straight, right, left]

        # Explore vs Exploit
        if random.randint(0, INITIAL_EPSILON_GAMES) < final_threshold:
            # Explore: choose a random move
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
            # print(f"Game {self.n_games}: Explore (Threshold: {final_threshold:.1f})") # Debug
        else:
            # Exploit: predict the best move using the model
            # Convert state (numpy array) to tensor for the model
            state_tensor = torch.tensor(state, dtype=torch.float, device=TORCH_DEVICE)
            # Add batch dimension if it's not already there (should be (1, C, H, W))
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)

            # Use the main model for action selection
            self.model.eval() # Set model to evaluation mode for prediction
            with torch.no_grad(): # No need for gradients during inference
                prediction = self.model(state_tensor)
            self.model.train() # Set model back to training mode

            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1
            # print(f"Game {self.n_games}: Exploit (Threshold: {final_threshold:.1f})") # Debug


        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    # Instantiate game first to pass to agent for dimensions
    game = SnakeGameAI()
    agent = Agent(game) # Pass game instance

    # --- Load Record Score ---
    record_file = './model/record.txt'
    if os.path.exists(record_file):
        try:
            with open(record_file, 'r') as f:
                record = int(f.read())
            print(f"Loaded record score: {record}")
        except ValueError:
            print("Could not read record score from file.")
            record = 0 # Ensure record is initialized
    # --- Load Record Score End ---


    while True:
        # get old state (image)
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (on the single step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember the experience for long memory training
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory() # Train on a batch from memory

            if score > record:
                record = score
                agent.model.save() # Save the CNN model
                # --- Save Record Score ---
                try:
                    with open(record_file, 'w') as f:
                        f.write(str(record))
                except IOError:
                     print(f"Could not write record score to {record_file}")
                # --- Save Record Score End ---

            # Optional: Save periodically regardless of record
            # if agent.n_games % 500 == 0: # Save every 500 games, for example
            #    agent.model.save(f'cnn_model_game_{agent.n_games}.pth')

            # Calculate current epsilon for printing
            current_epsilon_threshold = agent.epsilon_threshold - agent.n_games
            min_threshold = INITIAL_EPSILON_GAMES * MIN_EPSILON_GAMES_FACTOR
            final_threshold = max(min_threshold, current_epsilon_threshold)
            # Approximate epsilon probability (for display purposes)
            epsilon_prob = final_threshold / INITIAL_EPSILON_GAMES

            print(f'Game {agent.n_games} Score {score} Record: {record} Epsilon: {epsilon_prob:.3f}')

            if IS_PLOT_RENDERED:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()