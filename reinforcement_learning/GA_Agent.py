import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
from deap import base, creator, tools, algorithms   #引入遗传算法库
import random 
import numpy as np 
import functools
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.pre1_0 = nn.Linear(8,128)
        self.pre1_1 = nn.Linear(128,256) #256
        self.pre2_0 = nn.Linear(8,128)
        self.pre2_1 = nn.Linear(128,256) # 256
        self.pre3_0 = nn.Linear(12,128)
        self.pre3_1 = nn.Linear(128,256) # 256
        self.pre4_0 = nn.Linear(5,32) #32
        self.pre5_0 = nn.Linear(9,128)
        self.pre5_1 = nn.Linear(128,256) # 256
        self.pre6_0 = nn.Linear(14,128)
        self.pre6_1 = nn.Linear(128,256) # 256b

        self.fc1 = nn.Linear(1312, 4096)
        self.fc2 = nn.Linear(4096,512)
        self.fc3 = nn.Linear(512,19)

        
        self.activate_func = [nn.ReLU(), nn.Tanh()][1]  # Trick10: use tanh



    def forward(self, s):
        # print(s)
        s1 = s[0][0:8]
        # print(s1)
        s1 = self.pre1_0(s1)
        s1 = self.activate_func(s1)
        s1 = self.pre1_1(s1)
        s1 = self.activate_func(s1)

        s2 = s[0][8:16]
        s2 = self.pre2_0(s2)
        s2 = self.activate_func(s2)
        s2 = self.pre2_1(s2)
        s2 = self.activate_func(s2)

        s3 = s[0][16:28]
        s3 = self.pre3_0(s3)
        s3 = self.activate_func(s3)
        s3 = self.pre3_1(s3)
        s3 = self.activate_func(s3)
        
        s4 = s[0][28:33]
        s4 = self.pre4_0(s4)
        s4 = self.activate_func(s4)
        
        s5 = s[0][33:42]
        s5 = self.pre5_0(s5)
        s5 = self.activate_func(s5)
        s5 = self.pre5_1(s5)
        s5 = self.activate_func(s5)
        
        s6 = s[0][42:56]
        s6 = self.pre6_0(s6)
        s6 = self.activate_func(s6)
        s6 = self.pre6_1(s6)
        s6 = self.activate_func(s6)
        
        s1 = s1.unsqueeze(0)
        s2 = s2.unsqueeze(0)
        s3 = s3.unsqueeze(0)
        s4 = s4.unsqueeze(0)
        s5 = s5.unsqueeze(0)
        s6 = s6.unsqueeze(0)
        
        
        
        s_pre =  torch.cat((s1, s2, s3, s4,s5 , s6), dim=1)
        # print(s_pre.shape)
        s = self.fc1(s_pre)
        s = self.activate_func(s)
        s = self.fc2(s)
        s = self.activate_func(s)
        

        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob
        # print(a_prob)
        # return a_prob
    


class GA_net_discrete():
    def __init__(self): # 构造函数，只会被执行一次！
        # 声明遗传算法运行所需的基本参数
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化适应度
        # creator.create("Individual", list, fitness=creator.FitnessMin) # 使用list装入个体
        
        # # 只添加Actor网络使用的优化器
        # self.actor_GA_optimizer = base.Toolbox() # 创建针对Actor网络的优化器
        
        # # 定义Actor和Critic优化器的当前保存的种群和名人堂
        # self.actor_running_pop = None   #这里用来保存当前的种群
        
        # self.actor_best_pop = tools.HallOfFame(1)
        
        # # 定义统计量，输出遗传算法运行中的数据
        # self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        # self.stats.register("avg", np.mean)
        # self.stats.register("std", np.std)
        # self.stats.register("min", np.min)
        # self.stats.register("max", np.max)
        self.actor = Actor()
    
    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    
    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def count_model_params(self,model):  #获取一个模型中的参数数目
        model_state_dict = model.state_dict()
        count = 0
        for layer in model_state_dict:
            #print(model_state_dict[layer].shape)
            if len(model_state_dict[layer].shape) == 2:
                count += int(model_state_dict[layer].shape[0]) * int(model_state_dict[layer].shape[1])
            if len(model_state_dict[layer].shape) == 1:
                count += int(model_state_dict[layer].shape[0])
        return count 
    
    def load_model_with_list(self,model,weight): # 使用一个list装填模型
        now_index = 0
        model_state_dict = model.state_dict()
        for layer in model_state_dict:
            #print(model_state_dict[layer].shape)
            
            if len(model_state_dict[layer].shape) == 2:
                this_layer_count = int(model_state_dict[layer].shape[0]) * int(model_state_dict[layer].shape[1])
                model_state_dict[layer] = torch.tensor(weight[now_index:now_index + this_layer_count]).view(model_state_dict[layer].shape[0],model_state_dict[layer].shape[1])
            if len(model_state_dict[layer].shape) == 1:
                this_layer_count = int(model_state_dict[layer].shape[0])
                model_state_dict[layer] = torch.tensor(weight[now_index:now_index + this_layer_count]).view(-1)
                
            now_index += this_layer_count
            # print(this_layer_count)
        model.load_state_dict(model_state_dict)
        return 
    
    
