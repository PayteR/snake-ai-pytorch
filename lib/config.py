from lib.dotenvload import os
import torch

IS_PLOT_RENDERED=os.getenv('IS_PLOT_RENDERED').lower() == 'true'
IS_GAME_RENDERED=os.getenv('IS_GAME_RENDERED').lower() == 'true'

TORCH_DEVICE=os.getenv('DEVICE', 'cuda') 
if 'cuda' in TORCH_DEVICE and not torch.cuda.is_available():
    print('CUDA not available, switching to CPU')
    TORCH_DEVICE = 'cpu'
    
BLOCK_SIZE = int(os.getenv('BLOCK_SIZE', 20))
SPEED = int(os.getenv('SPEED', 4000000))
MAX_MEMORY = int(os.getenv('MAX_MEMORY', 100_000))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))
LR = float(os.getenv('LR', 0.001))
EPSILON = int(os.getenv('EPSILON', 80))