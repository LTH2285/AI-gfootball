import torch
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as football_env
from utils import calculating_state

from GA_Agent import GA_net_discrete
import matplotlib.pyplot as plt 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from deap import base, creator, tools, algorithms   #引入遗传算法库

agent = GA_net_discrete()

def create_single_football_env(iprocess):
      # """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name='1_vs_1_easy', stacked='stacked',
      rewards='scoring',
      logdir=logger.get_dir(),
      representation='simple115v2',
  )
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  # env = gym.make('GFootball-1_vs_1_easy_stochastic-SMM-v0', representation='simple115')
  return env

def evaluate_policy():
    
    agent.actor.load_state_dict(torch.load('./gfootball/examples/model_weights.pth'))
    
    # state_norm = Normalization(shape=3)
    # env = gym.make("Pendulum-v1")
    env = create_single_football_env(1)
    times = 1
    evaluate_reward = 0
    
    for _ in range(times):
        
        s = env.reset()
        _ , slist = calculating_state(s)

        # s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            env.render()
            a = agent.evaluate(slist)  # We use the deterministic policy during the evaluating
            action = a
            s_, r, done, _ = env.step(action)
            _ , s_list = calculating_state(s_)
            # s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            slist = s_list
            time.sleep(0.05)
        evaluate_reward += episode_reward
    res = evaluate_reward / times
    res = [res]
    return res

evaluate_policy()

