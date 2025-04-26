from lib.dotenvload import os
import torch

IS_PLOT_RENDERED=os.getenv('IS_PLOT_RENDERED', 'false').lower() == 'true'
IS_GAME_RENDERED=os.getenv('IS_GAME_RENDERED', 'false').lower() == 'true'

TORCH_DEVICE=os.getenv('DEVICE', 'cuda')
if 'cuda' in TORCH_DEVICE and not torch.cuda.is_available():
    print('CUDA not available, switching to CPU')
    TORCH_DEVICE = 'cpu'
elif 'cuda' in TORCH_DEVICE:
    print(f'Using CUDA device: {torch.cuda.get_device_name(0)}') # Print GPU name

BLOCK_SIZE = int(os.getenv('BLOCK_SIZE', 20))
SPEED = int(os.getenv('SPEED', 40)) # Reduce speed if rendering, increase if not rendering for faster training

# Hyperparameters - adjust as needed for CNN
MAX_MEMORY = int(os.getenv('MAX_MEMORY', 100_000)) # Keep large or increase if VRAM allows
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000)) # Can be large with 24GB VRAM, but test stability. Maybe 512 or 1024.
LR = float(os.getenv('LR', 0.00025)) # Lower learning rate for CNN stability
EPSILON = int(os.getenv('EPSILON', 150)) # Number of games for initial linear epsilon decay phase