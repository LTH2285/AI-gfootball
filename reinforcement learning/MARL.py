# 这是一份实现多智能体强化学习的程序
# 基本想法是 每个智能体都采用相同的网络进行推理，可以观测到球场上的部分信息； 中心的策略网络则
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as football_env
import gym 
import numpy as np 
from utils import calculating_state , calculating_reward , calculating_MA_states
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from MAPPO_discrete import PPO_discrete
import matplotlib.pyplot as plt 
import torch
import time 
import pandas as pd  


def create_multi_football_env(iprocess):
  # """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name='3v3', stacked='stacked',
      rewards='scoring,checkpoints',
      logdir=logger.get_dir(),
      representation='simple115v2',
      number_of_left_players_agent_controls = 3,
  )
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)),allow_early_resets= True )
  # env = gym.make('GFootball-1_vs_1_easy_stochastic-SMM-v0', representation='simple115')
  return env

env = create_multi_football_env(1) #创建环境
env_evaluate = create_multi_football_env(1)

# env = gym.make("CartPole-v0")
# env_evaluate = gym.make("CartPole-v0")



def evaluate_policy(args, env, agent, state_norm):
    times = 2
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        sdict,tocritic,toagent1,toagent2,toagent3 = calculating_MA_states(s) # 对初态计算一次
        if args.use_state_norm:  # During the evaluating,update=False
            toagent1 = state_norm(toagent1, update=False)
            toagent2 = state_norm(toagent2, update=False)
            toagent3 = state_norm(toagent3, update=False)

        done = False
        episode_reward = 0
        episode_step = 0
        while not done:
            # sdict,slist = calculating_state(s)
            a_agent1 = agent.evaluate(toagent1)  # We use the deterministic policy during the evaluating
            a_agent2 = agent.evaluate(toagent2)
            a_agent3 = agent.evaluate(toagent3)
            
            s_, r, done, _ = env.step([a_agent1,a_agent2,a_agent3]) #传入动作，获取次态
            sdict_,tocritic_,toagent1_,toagent2_,toagent3_ = calculating_MA_states(s_)
            
            if args.use_state_norm:
                toagent1_ = state_norm(toagent1_, update=False)
                toagent2_ = state_norm(toagent2_, update=False)
                toagent3_ = state_norm(toagent3_, update=False)
            episode_reward += r[0]+r[1]+r[2]
            tocritic = tocritic_
            toagent1 = toagent1_
            toagent2 = toagent2_
            toagent3 = toagent3_ 
            sdict = sdict_
            episode_step += 1
            if episode_step > 2950:
                done = True 
            
        evaluate_reward += episode_reward

    return evaluate_reward / times

def PPO_train(args, env):
  record_rewards = []
  record_smooth_rewards = []
  record_goals = []
  smooth_reward = 0
  episode_goal = 0
  episode = 0


  evaluate_num = 0  # Record the number of evaluations
  evaluate_rewards = []  # Record the rewards during the evaluating
  evaluate_episodes = []
  total_steps = 0  # Record the total steps during the training

  replay_buffer = ReplayBuffer(args)
  agent = PPO_discrete(args)

################################################################################
#   saved_parameters_actor = torch.load('./gfootball/examples/MAactor_parameters.pth')

#   # # 将加载的参数赋值给新创建的网络实例
#   agent.actor.load_state_dict(saved_parameters_actor)

#   saved_parameters_critic = torch.load('./gfootball/examples/MAcritic_parameters.pth')

#   # # 将加载的参数赋值给新创建的网络实例
#   agent.critic.load_state_dict(saved_parameters_critic)
##################################################################################

  critic_state_norm = Normalization(shape = 54)  # Trick 2:state normalization #给critic用的标准化
  actor_state_norm = Normalization(shape = 26) #给Actor使用的标准化
  
  if args.use_reward_norm:  # Trick 3:reward normalization
      reward_norm = Normalization(shape=1)
  elif args.use_reward_scaling:  # Trick 4:reward scaling
      reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

  # 记录数据
  data = []        



  while total_steps < args.max_train_steps:
    episode += 1
    s = env.reset()
    # print(s)
    sdict,tocritic,toagent1,toagent2,toagent3 = calculating_MA_states(s) #计算状态字典、发送给critic的全局状态以及发送给每个智能体的状态
    if args.use_state_norm:
        tocritic = critic_state_norm(tocritic) #critic和Agent分别进行标准化
        toagent1 = actor_state_norm(toagent1)
        toagent2 = actor_state_norm(toagent2)
        toagent3 = actor_state_norm(toagent3)
        # s = state_norm(s)
    if args.use_reward_scaling:
        reward_scaling.reset()
    episode_steps = 0
    episode_reward = 0
    episode_goal = 0


    # episode_data = np.array([0.0,0.0,0,0,0,0])

    done = False
    while not done and episode_steps < 2950: #这里需要提前结束，不然环境要出问题

      episode_steps += 1

    # 接下来，三个智能体将依次选择动作
    #   print(toagent1)
      a_agent1, logprob_agent1 = agent.choose_action(toagent1)  #智能体1选择动作，并计算动作的对数概率
      a_agent2, logprob_agent2 = agent.choose_action(toagent2)  #智能体2再选择动作
      a_agent3, logprob_agent3 = agent.choose_action(toagent3)
      # a = agent.evaluate(slist)
      s_, r, done, _ = env.step([a_agent1,a_agent2,a_agent3])
      sdict_,tocritic_,toagent1_,toagent2_,toagent3_ = calculating_MA_states(s_)

      r = r[0] + r[1] + r[2] #这里，总体的奖励计算为：直接将三个分奖励加起来
      episode_goal += r

      episode_reward += r #暂且使用自带的奖励


      if args.use_state_norm:

        tocritic_ = critic_state_norm(tocritic_) #critic和Agent分别进行标准化，这里标准化的是次态
        toagent1_ = actor_state_norm(toagent1_)
        toagent2_ = actor_state_norm(toagent2_)
        toagent3_ = actor_state_norm(toagent3_)
      if args.use_reward_norm:
          r = reward_norm(r)
      elif args.use_reward_scaling:
          r = reward_scaling(r)


      # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
      # dw means dead or win,there is no next state s';
      # but when reaching the max_episode_steps,there is a next state s' actually.
      if done and episode_steps != 10000:
          dw = True
      else:
          dw = False
      if episode_steps == 2949:
          done = True 

      # Take the 'action'，but store the original 'a'（especially for Beta）
      replay_buffer.store(tocritic, a_agent1, logprob_agent1, r, tocritic_, dw, done) #向buffer内存入状态的时候，实际上应该存critic可以读取到的全局状态。因为过会critic要根据全局状态评价智能体们的动作
      replay_buffer.store(tocritic, a_agent2, logprob_agent2, r, tocritic_, dw, done)
      replay_buffer.store(tocritic, a_agent3, logprob_agent3, r, tocritic_, dw, done)
      # 所有智能体共享同一个critic，所以所有智能体的东西都可以存进一个replaybuffer中
        
      tocritic = tocritic_
      toagent1 = toagent1_
      toagent2 = toagent2_
      toagent3 = toagent3_ 
      sdict = sdict_ # 更新一遍次态
      
      total_steps += 1

      # When the number of transitions in  buffer reaches batch_size,then update
      if replay_buffer.count >= args.batch_size :
          agent.update(replay_buffer, total_steps)
          replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
      if total_steps % args.evaluate_freq == 0:
        evaluate_num += 1
        evaluate_reward = evaluate_policy(args, env_evaluate, agent, actor_state_norm)
        evaluate_rewards.append(evaluate_reward)
        print(
            f"==========evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward}========== \t"
        )
        # evaluates.append(episode_reward)
        evaluate_episodes.append(episode)
                      # Save the rewards

    # episode_data = episode_data.tolist()
    # data.append(episode_data)
    record_goals.append(episode_goal)
    record_rewards.append(episode_reward)
    print("episode:  ",episode,"reward:  ",episode_reward,"goals:  ",episode_goal)
    if episode == 1:
        smooth_reward = episode_reward 
    else :
        smooth_reward = smooth_reward * 0.95 + episode_reward * 0.05
    record_smooth_rewards.append(smooth_reward)

    if episode % 4 == 0 :
        plt.plot(record_rewards,label = "reward",color = "silver")
        plt.plot(record_smooth_rewards,label = "smooth",color = "royalblue",linewidth = 2)
        plt.plot(evaluate_episodes,evaluate_rewards,label = "evaluate",color = "tomato",linewidth = 2)
        # print(evaluates)
        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig('./gfootball/examples/MAresult.png',dpi = 1000)
        plt.close()
        plt.plot(record_goals,label = "goals",color = "tomato",linewidth = 2)
        plt.xlabel("Episode")
        plt.ylabel("Goals")
        plt.savefig('./gfootball/examples/MAgoal.png',dpi = 1000)
        plt.close()

    if episode % 30 == 0:
        # print("saving")
        torch.save(agent.actor.state_dict(), './gfootball/examples/MAactor_parameters.pth')
        torch.save(agent.critic.state_dict(), './gfootball/examples/MAcritic_parameters.pth')
        # df = pd.DataFrame(data,columns=['reward_goal' , 'reward_dis' , 'reward_con' , 'reward_dir' , 'reward_atk' , 'reward_ball_dir'])
        # df.to_csv("./gfootball/examples/PPO_data.csv") 

  plt.plot(record_rewards,label = "reward",color = "silver")
  plt.plot(record_smooth_rewards,label = "smooth",color = "royalblue",linewidth = 2)
  plt.legend()
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.savefig('./gfootball/examples/result.png',dpi = 1000)
  plt.show()

  plt.plot(record_goals,label = "goals",color = "tomato",linewidth = 2)
  plt.xlabel("Episode")
  plt.ylabel("Goals")
  plt.savefig('./gfootball/examples/goal.png',dpi = 1000)
  plt.show()
    
    

def main():
    PPO_train(args,env)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(50e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=9e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=18000, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1024, help="Minibatch size")
    # parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=6e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=6e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.015, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    args.state_dim = 54
    args.action_dim = 19
    
    
    PPO_train(args,env)
    # for i in range(10):
    # print(play_a_episode_randomly(env))
    
    