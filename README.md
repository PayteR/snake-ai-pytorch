# this repo is forked from:

https://github.com/patrickloeber/snake-ai-pytorch

I reworked code:
* deadend trap prediction (not working good yet), check `is_collision_predict` function in `game.py` file
* to save model after record or every 100th run
* to load model from file by default
* add `.env` file to set parameters
* add `requirements.txt` file
* `cuda` support

# Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to setup the environment and implement the Snake game.
- Part 3: Implement the agent that controls the game.
- Part 4: Implement the neural network to predict the moves and train it.
