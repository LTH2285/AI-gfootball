"""
Author: LTH
Date: 2023-05-31 11:56:29
LastEditTime: 2023-05-31 16:07:05
FilePath: \files\python_work\课程\artificial_inteligence\base_on_rules.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved. 

"""
import gfootball.env as football_env
import utils
import rules
from gym.wrappers import Monitor


# import multiprocessing
import os

# from absl import app
# from absl import flags
from baselines import logger

from baselines.bench import monitor

# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.ppo2 import ppo2
import gfootball.env as football_env

# import gym
import numpy as np

# from utils import calculating_state, calculating_reward

# import argparse
# from normalization import Normalization, RewardScaling
# from replaybuffer import ReplayBuffer
# from ppo_discrete import PPO_discrete
import matplotlib.pyplot as plt

# import torch
# import time
# import pandas as pd


# 创建gfootball环境
def create_single_football_env(iprocess):
    # """Creates gfootball environment."""
    env = football_env.create_environment(
        env_name="1_vs_1_easy",
        # env_name="shabi",
        stacked="stacked",
        # rewards="scoring,checkpoints",
        rewards="scoring",
        logdir=logger.get_dir(),
        representation="simple115v2",
    )
    env = monitor.Monitor(
        env,
        logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess)),
        allow_early_resets=True,
    )
    # env = Monitor(env, 'logs/', allow_early_resets=True)
    # env = gym.make('GFootball-1_vs_1_easy_stochastic-SMM-v0', representation='simple115')
    return env


env = create_single_football_env(1)  # 创建环境
clear = False
clear_count = 0
wining_count = 0


count_r = 0
count_sum = np.asarray([0, 0, 0])
for episode in range(500):
    s = env.reset()
    sdict, slist = utils.calculating_state(s)
    done = False
    count = np.asarray([0, 0, 0])
    episode_reward = 0
    while episode_reward == 0:  # while not done:
        # env.render()
        if sdict["control"][1]:
            # act = rules.make_decision_with_ball(rules.get_id(sdict))
            act = rules.make_decision_with_ball(sdict)

            # act = rules.make_decision_without_ball(sdict)
            # print("control")
            count[0] += 1
        else:
            act = rules.make_decision_without_ball(sdict)
            if sdict["control"][2]:
                count[1] += 1
            else:
                count[2] += 1
            if not clear:
                act = 14
                clear = True
                clear_count = 0
            else:
                clear_count += 1
                if clear_count == 15:
                    clear = False
        # print(act)
        if act is None:
            act = 5
        s, r, done, _ = env.step(act)
        episode_reward += r
        sdict, slist = utils.calculating_state(s)
        if done:
            break
    if episode_reward > 0:
        wining_count += 1
    print("episode", episode, "reward", episode_reward)
    print("control percentage:", 100 * count / np.sum(count))
    count_sum += count
print(f"wining percentage:{wining_count/5}%")
print("average control percentage:", 100 * count_sum / np.sum(count_sum))
