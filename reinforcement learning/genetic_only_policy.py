import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from utils import calculating_reward , calculating_state
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as football_env
import gfootball

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
      rewards='scoring,checkpoints',
      logdir=logger.get_dir(),
      representation='simple115v2',
  )
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  # env = gym.make('GFootball-1_vs_1_easy_stochastic-SMM-v0', representation='simple115')
  return env

env = create_single_football_env(1) #创建环境


def evaluate_policy(individual):
    agent.load_model_with_list(agent.actor,individual) # 加入新的网络权值
    # state_norm = Normalization(shape=17)
    times = 5
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        sdict,slist = calculating_state(s)
          # During the evaluating,update=False
        # slist = state_norm(slist, update=False)
        done = False
        episode_reward = 0
        while not done:
            # sdict,slist = calculating_state(s)
            a = agent.evaluate(slist)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            s_dict,s_list = calculating_state(s_)
            # r , _ = calculating_reward(s_dict,r)
            
            # s_list = state_norm(s_list, update=False)
            episode_reward += r
            slist = s_list
            sdict = s_dict
        evaluate_reward += episode_reward

    return [evaluate_reward / times]




# 声明遗传算法运行所需的基本参数
creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # 最大化适应度
creator.create("Individual", list, fitness=creator.FitnessMin) # 使用list装入个体
        
# 只添加Actor网络使用的优化器
actor_GA_optimizer = base.Toolbox() # 创建针对Actor网络的优化器
        
# 定义Actor优化器的当前保存的种群和名人堂
actor_running_pop = None   #这里用来保存当前的种群
        
actor_best_pop = tools.HallOfFame(1)
        
# 定义统计量，输出遗传算法运行中的数据
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
        
# 注册Actor的GA优化器的一系列机制
actor_num_parameters = agent.count_model_params(agent.actor)
actor_GA_optimizer.register("attr_float", random.uniform, -2, 2)  # 属性生成器（随机浮点数）
actor_GA_optimizer.register("individual", tools.initRepeat, creator.Individual,  actor_GA_optimizer.attr_float, n = actor_num_parameters)  # 个体运算符
actor_GA_optimizer.register("population", tools.initRepeat, list, actor_GA_optimizer.individual)  # 种群运算符 
actor_GA_optimizer.register("mate", tools.cxBlend,alpha = 0.5)  # 交叉运算符
actor_GA_optimizer.register("mutate", tools.mutGaussian, mu = 0,sigma = 0.2,indpb = 0.10)  # 变异运算符
actor_GA_optimizer.register("select", tools.selTournament, tournsize=6)  # 选择运算符
actor_GA_optimizer.register("evaluate", evaluate_policy)  
actor_running_pop = actor_GA_optimizer.population(n = 20)


    
def main():
    global actor_running_pop
    actor_running_pop, logbook = algorithms.eaSimple(actor_running_pop, actor_GA_optimizer, cxpb=0.85, mutpb=0.10, ngen=20, stats=stats, halloffame=actor_best_pop, verbose=True)
    
    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")

    # Plot the statistics
    plt.plot(gen, min_, label="Minimum",color = "silver")
    plt.plot(gen, avg, label="Average",color = "royalblue",linewidth = 3)
    plt.plot(gen, max_, label="Max",color = "silver")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
    
    agent.load_model_with_list(agent.actor,actor_best_pop[0])
    torch.save(agent.actor.state_dict(), './gfootball/examples/model_weights.pth')
    
    return 

main()
