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
from utils import calculating_state, calculating_reward
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
import matplotlib.pyplot as plt
import torch
import time
import pandas as pd


def create_single_football_env(iprocess):
    # """Creates gfootball environment."""
    env = football_env.create_environment(
        env_name="1v1",
        stacked="stacked",
        rewards="scoring,checkpoints",
        logdir=logger.get_dir(),
        representation="simple115v2",
    )
    env = monitor.Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess))
    )
    # env = gym.make('GFootball-1_vs_1_easy_stochastic-SMM-v0', representation='simple115')
    return env


env = create_single_football_env(1)  # 创建环境
env_evaluate = create_single_football_env(1)

# env = gym.make("CartPole-v0")
# env_evaluate = gym.make("CartPole-v0")


def play_a_episode_randomly(env):
    env.reset()
    action = env.action_space.sample()  # 随机采样动作
    done = False
    episode_reward = 0
    while not done:
        # 渲染环境
        # env.render()
        observation, reward, done, info = env.step(action)
        dic, li = calculating_state(observation)
        reward, _ = calculating_reward(dic=dic, reward=reward)

        # print(len(li))
        print(dic)
        time.sleep(0.3)
        action = env.action_space.sample()
        episode_reward += reward
    return episode_reward


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        # agent.actor.GRUnet.reset_hidden_state(1)
        setzero = np.zeros(10).tolist()
        s = env.reset()
        sdict, slist = calculating_state(s)
        slist = slist + setzero
        if args.use_state_norm:  # During the evaluating,update=False
            slist = state_norm(slist, update=False)
            # s = state_norm(s)
        done = False
        episode_reward = 0
        while not done:
            # sdict,slist = calculating_state(s)
            a = agent.evaluate(
                slist
            )  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            s_dict, s_list = calculating_state(s_)
            s_list = s_list + setzero
            r, rlist = calculating_reward(s_dict, r)

            if args.use_state_norm:
                s_list = state_norm(s_list, update=False)
                # s_ = state_norm(s_)
            episode_reward += r
            slist = s_list
            sdict = s_dict
            s = s_
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
    smooth_evaluate_record = []
    smooth_evaluate = 0
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    noise = np.random.normal(0, 1, 10).tolist()

    #   saved_parameters_actor = torch.load('./gfootball/examples/actor_parameters.pth')

    #   # # 将加载的参数赋值给新创建的网络实例
    #   agent.actor.load_state_dict(saved_parameters_actor)

    #   saved_parameters_critic = torch.load('./gfootball/examples/critic_parameters.pth')

    #   # # 将加载的参数赋值给新创建的网络实例
    #   agent.critic.load_state_dict(saved_parameters_critic)

    state_norm = Normalization(shape=66)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # 记录数据
    data = []

    while total_steps < args.max_train_steps:
        episode += 1
        s = env.reset()
        sdic, slist = calculating_state(s)
        slist = slist + noise
        if args.use_state_norm:
            slist = state_norm(slist)
            # s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        episode_reward = 0
        episode_goal = 0

        # episode_data = np.array([0.0,0.0,0,0,0,0])
        # agent.actor.GRUnet.reset_hidden_state(1)
        done = False
        while not done:
            # env.render()
            # if episode % 2 ==0:
            #     time.sleep(1)
            episode_steps += 1
            # sdic,slist = calculating_state(s)

            a, a_logprob = agent.choose_action(
                slist
            )  # Action and the corresponding log probability

            # a = agent.evaluate(slist)
            s_, r, done, _ = env.step(a)
            s_dic, s_list = calculating_state(s_)
            s_list = s_list + noise
            #   print(s_list)
            # print(s_list)

            episode_goal += r
            r, rlist = calculating_reward(s_dic, r)
            episode_reward += r

            # rlist = np.array(rlist)
            # episode_data += rlist

            if args.use_state_norm:
                # s_list = state_norm(s_list)
                s_list = state_norm(s_list)
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

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(slist, a, a_logprob, r, s_list, dw, done)
            # s = s_
            slist = s_list
            sdic = s_dic
            total_steps += 1

            # When the number of transitions in  buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print(
                    f"==========evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward}========== \t"
                )
                # evaluates.append(episode_reward)
                evaluate_episodes.append(episode)
                # Save the rewards

        # episode_data = episode_data.tolist()
        # data.append(episode_data)

        # agent.update(replay_buffer, total_steps)
        # replay_buffer.count = 0

        record_goals.append(episode_goal)
        record_rewards.append(episode_reward)
        print(
            "episode:  ", episode, "reward:  ", episode_reward, "goals:  ", episode_goal
        )
        if episode == 1:
            smooth_reward = episode_reward
        else:
            smooth_reward = smooth_reward * 0.95 + episode_reward * 0.05
        record_smooth_rewards.append(smooth_reward)

        # print(len(evaluate_episodes))
        if not evaluate_episodes:
            smooth_evaluate = smooth_reward
        elif len(evaluate_episodes) == 1:
            smooth_evaluate = evaluate_rewards[-1]
        else:
            smooth_evaluate = smooth_evaluate * 0.95 + evaluate_rewards[-1] * 0.05
        smooth_evaluate_record.append(smooth_evaluate)

        if episode % 5 == 0:
            plt.plot(
                record_rewards,
                label="reward",
                color="royalblue",
                linewidth=1,
                alpha=0.3,
            )
            plt.plot(
                record_smooth_rewards, label="smooth", color="royalblue", linewidth=2
            )
            plt.plot(
                evaluate_episodes,
                evaluate_rewards,
                label="evaluate",
                color="tomato",
                linewidth=1,
                alpha=0.3,
            )
            plt.plot(
                smooth_evaluate_record, label="smooth", color="tomato", linewidth=2
            )
            # print(evaluates)
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig("./gfootball/examples/result.png", dpi=1000)
            plt.close()
            plt.plot(record_goals, label="goals", color="tomato", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Goals")
            plt.savefig("./gfootball/examples/goal.png", dpi=1000)
            plt.close()

        if episode % 100 == 0:
            # print("saving")
            torch.save(
                agent.actor.state_dict(), "./gfootball/examples/actor_parameters.pth"
            )
            torch.save(
                agent.critic.state_dict(), "./gfootball/examples/critic_parameters.pth"
            )
            # df = pd.DataFrame(data,columns=['reward_goal' , 'reward_dis' , 'reward_con' , 'reward_dir' , 'reward_atk' , 'reward_ball_dir'])
            # df.to_csv("./gfootball/examples/PPO_data.csv")

        if episode % 10 == 0:
            noise = np.random.normal(0, 1, 10).tolist()

    plt.plot(record_rewards, label="reward", color="silver")
    plt.plot(record_smooth_rewards, label="smooth", color="royalblue", linewidth=2)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./gfootball/examples/result.png", dpi=1000)
    plt.show()

    plt.plot(record_goals, label="goals", color="tomato", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Goals")
    plt.savefig("./gfootball/examples/goal.png", dpi=1000)
    plt.show()


def main():
    PPO_train(args, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(50e5),
        help=" Maximum number of training steps",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=3e3,
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=501, help="Batch size")
    parser.add_argument(
        "--mini_batch_size", type=int, default=501, help="Minibatch size"
    )
    # parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument(
        "--lr_a", type=float, default=3e-4, help="Learning rate of actor"
    )
    parser.add_argument(
        "--lr_c", type=float, default=5e-4, help="Learning rate of critic"
    )
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument(
        "--epsilon", type=float, default=0.10, help="PPO clip parameter"
    )
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument(
        "--use_adv_norm",
        type=bool,
        default=True,
        help="Trick 1:advantage normalization",
    )
    parser.add_argument(
        "--use_state_norm", type=bool, default=True, help="Trick 2:state normalization"
    )
    parser.add_argument(
        "--use_reward_norm",
        type=bool,
        default=False,
        help="Trick 3:reward normalization",
    )
    parser.add_argument(
        "--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.010, help="Trick 5: policy entropy"
    )
    parser.add_argument(
        "--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay"
    )
    parser.add_argument(
        "--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip"
    )
    parser.add_argument(
        "--use_orthogonal_init",
        type=bool,
        default=True,
        help="Trick 8: orthogonal initialization",
    )
    parser.add_argument(
        "--set_adam_eps",
        type=float,
        default=True,
        help="Trick 9: set Adam epsilon=1e-5",
    )
    parser.add_argument(
        "--use_tanh",
        type=float,
        default=True,
        help="Trick 10: tanh activation function",
    )

    args = parser.parse_args()
    args.state_dim = 66
    args.action_dim = 19

    PPO_train(args, env)
    # for i in range(10):
    # print(play_a_episode_randomly(env))
