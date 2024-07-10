import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import os
from breakout import *
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device)

model = AtariNet(nb_actions=4)

model.to(device)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=0.16055330662888903,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.00001,
              memory_capacity=1000000,
              batch_size=64)

old_stats = {}
try:
   old_stats = np.load('stats.npy', allow_pickle=True).item()
   print(f"Succesfully loaded stats file")
except:
   print(f"No stats file available")

agent.train(env=environment, epochs=200000, old_stats=old_stats)