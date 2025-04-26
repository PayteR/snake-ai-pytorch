import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from lib.config import TORCH_DEVICE, LR # Assuming LR is needed here, otherwise remove
import os
import copy # Import copy

TARGET_UPDATE_FREQUENCY = 100 # Update target network every 100 training steps (can be tuned)
MODEL_FILENAME = 'cnn_model.pth' # New model filename

# Define the CNN architecture
class Conv_QNet(nn.Module):
    def __init__(self, input_channels, h, w, output_size):
        super().__init__()
        # Input shape: (Batch, Channels, Height, Width) e.g. (B, 1, 24, 32) for 480x640 game, 20 block size
        # Or (B, 4, H, W) if using frame stacking

        # Convolutional layers (example architecture, can be tuned)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2, device=TORCH_DEVICE) # Keep spatial dims
        self.bn1 = nn.BatchNorm2d(32, device=TORCH_DEVICE)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=TORCH_DEVICE) # Keep spatial dims
        self.bn2 = nn.BatchNorm2d(64, device=TORCH_DEVICE)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, device=TORCH_DEVICE) # Keep spatial dims
        self.bn3 = nn.BatchNorm2d(64, device=TORCH_DEVICE)

        # Function to calculate flattened size after conv layers
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            # Formula: floor(((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)
            # Simplified for dilation=1
             return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        # Calculate the output size of the conv layers dynamically
        # Note: This assumes padding maintains size for stride=1 kernels used here.
        # If using different kernels/strides/padding, adjust this calculation.
        convw = w # conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 5, 1, 2), 3, 1, 1), 3, 1, 1)
        convh = h # conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 5, 1, 2), 3, 1, 1), 3, 1, 1)
        linear_input_size = 64 * convw * convh # 64 filters from conv3

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512, device=TORCH_DEVICE) # Large hidden layer
        self.fc2 = nn.Linear(512, output_size, device=TORCH_DEVICE) # Output layer (actions)

    def forward(self, x):
        # Apply conv layers with ReLU activation and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation on the final output layer (Q-values)
        return x

    def save(self, file_name=MODEL_FILENAME):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        # print(f"Saved model state to {file_name}") # Optional confirmation

    def load(self, file_name=MODEL_FILENAME):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            try:
                self.load_state_dict(torch.load(file_name, map_location=TORCH_DEVICE))
                print(f"Loaded model state from {file_name}")
            except Exception as e:
                print(f"Error loading model state from {file_name}: {e}. Check model architecture compatibility.")
        else:
            print(f"Model file not found at {file_name}, starting with initial weights.")


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target network is only for inference

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Consider Huber loss (SmoothL1Loss) which is less sensitive to outliers than MSELoss
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        self.train_step_counter = 0

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors on the correct device
        # State and next_state are now potentially image tensors (N, C, H, W)
        state = torch.tensor(np.array(state), dtype=torch.float, device=TORCH_DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=TORCH_DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long, device=TORCH_DEVICE) # Action indices/vectors
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=TORCH_DEVICE)
        done = np.array(done) # Keep done as numpy array for easier indexing

        # Handle single experience vs batch
        if len(state.shape) == 3: # Single image (C, H, W)
            # Add batch dimension (1, C, H, W)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # Keep as iterable

        # Ensure actions are in the correct format (batch_size, num_actions) if they are one-hot
        # Or (batch_size,) if they are indices. Assuming one-hot [1,0,0] format from agent.
        # If actions are indices, need to adjust how target is updated.
        # Let's assume actions are passed as one-hot encoded vectors [0,1,0] etc.
        # We need the index of the action taken.
        action_indices = torch.argmax(action, dim=1) # Get index (0, 1, or 2) for each action in batch

        # 1: predicted Q values with current state from main model
        pred = self.model(state) # Shape: (batch_size, 3)

        # 2: Q_new = r + y * max(next_predicted Q value from target_model)
        target = pred.clone()
        with torch.no_grad(): # No gradients needed for target calculation
            next_pred = self.target_model(next_state) # Shape: (batch_size, 3)
            max_next_pred = torch.max(next_pred, dim=1)[0] # Get max Q value for each next state

        # Calculate target Q values
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max_next_pred[idx]

            # Update the Q value for the action actually taken
            target[idx][action_indices[idx]] = Q_new

        # 3. Calculate loss and optimize
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        # Optional: Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        self.optimizer.step()

        # 4. Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_model()

    def update_target_model(self):
        # print("Updating target model") # Optional: for debugging
        self.target_model.load_state_dict(self.model.state_dict())

    # Add methods to save/load trainer state if needed (optimizer state, step counter)
    # def save_trainer_state(self, path): ...
    # def load_trainer_state(self, path): ...
