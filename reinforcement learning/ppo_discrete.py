import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorGRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ActorGRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 1)
        self.hidden_state = None

    def reset_hidden_state(
        self,
        batchsize,
    ):
        self.hidden_state = torch.zeros(
            1, batchsize, self.hidden_size
        )  # 1层,batchsize,hiddensize

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加batchsize维度
        output, self.hidden_state = self.gru(x, self.hidden_state)
        output = output.squeeze(1)  # 减去batchsize维度

        return output

    def decide(self, x):
        with torch.no_grad():  # 在从环境中采样时，完全不需要梯度这件事情
            x = x.unsqueeze(1)  # 加上batchsize维度
            # print(x)
            # print(self.hidden_state)
            output, self.hidden_state = self.gru(x, self.hidden_state)  # 在不计算梯度的情况下正向传播
            # self.hidden_state = self.hidden_state.squeeze(0)
            output = output.squeeze(1)

            return output


class CriticGRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticGRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 1)
        self.hidden_state = None

    def reset_hidden_state(
        self,
        batchsize,
    ):
        self.hidden_state = torch.zeros(
            1, batchsize, self.hidden_size
        )  # 1层,batchsize,hiddensize

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加batchsize维度
        output, self.hidden_state = self.gru(x, self.hidden_state)
        output = output.squeeze(1)  # 减去batchsize维度

        return output

    def predict(self, x):
        with torch.no_grad():  # 在从环境中采样时，完全不需要梯度这件事情
            x = x.unsqueeze(1)  # 加上batchsize维度
            output, self.hidden_state = self.gru(x, self.hidden_state)  # 在不计算梯度的情况下正向传播
            output = output.squeeze(1)

            return output


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        # self.hidden_state = self.init_hidden_state()
        self.pre1_0 = nn.Linear(8, 128)
        self.pre1_1 = nn.Linear(128, 256)  # 256
        self.pre2_0 = nn.Linear(8, 128)
        self.pre2_1 = nn.Linear(128, 256)  # 256
        self.pre3_0 = nn.Linear(12, 128)
        self.pre3_1 = nn.Linear(128, 256)  # 256
        self.pre4_0 = nn.Linear(5, 32)  # 32
        self.pre5_0 = nn.Linear(9, 128)
        self.pre5_1 = nn.Linear(128, 256)  # 256
        self.pre6_0 = nn.Linear(14, 128)
        self.pre6_1 = nn.Linear(128, 256)  # 256
        self.pre7_0 = nn.Linear(10, 128)

        self.fc1 = nn.Linear(1312 + 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 19)

        self.GRUnet = CriticGRUNetwork(512, 256)

        self.activate_func = [nn.ReLU(), nn.Tanh()][1]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.pre1_0)
            orthogonal_init(self.pre1_1)
            orthogonal_init(self.pre2_0)
            orthogonal_init(self.pre2_1)
            orthogonal_init(self.pre3_0)
            orthogonal_init(self.pre3_1)
            orthogonal_init(self.pre4_0)
            orthogonal_init(self.pre5_0)
            orthogonal_init(self.pre5_1)
            orthogonal_init(self.pre6_0)
            orthogonal_init(self.pre6_1)
            orthogonal_init(self.fc3, gain=0.01)

    def init_hidden_state(self, length):
        return torch.zeros(length, 1, 2048)

    def forward(self, s):
        s1 = s[:, 0:8]
        # print(s1)
        s1 = self.pre1_0(s1)
        s1 = self.activate_func(s1)
        s1 = self.pre1_1(s1)
        s1 = self.activate_func(s1)

        s2 = s[:, 8:16]
        s2 = self.pre2_0(s2)
        s2 = self.activate_func(s2)
        s2 = self.pre2_1(s2)
        s2 = self.activate_func(s2)

        s3 = s[:, 16:28]
        s3 = self.pre3_0(s3)
        s3 = self.activate_func(s3)
        s3 = self.pre3_1(s3)
        s3 = self.activate_func(s3)

        s4 = s[:, 28:33]
        s4 = self.pre4_0(s4)
        s4 = self.activate_func(s4)

        s5 = s[:, 33:42]
        s5 = self.pre5_0(s5)
        s5 = self.activate_func(s5)
        s5 = self.pre5_1(s5)
        s5 = self.activate_func(s5)

        s6 = s[:, 42:56]
        s6 = self.pre6_0(s6)
        s6 = self.activate_func(s6)
        s6 = self.pre6_1(s6)
        s6 = self.activate_func(s6)

        s7 = s[:, 56:66]
        s7 = self.pre7_0(s7)
        s7 = self.activate_func(s7)

        # s1 = s1.unsqueeze(0)
        # s2 = s2.unsqueeze(0)
        # s3 = s3.unsqueeze(0)
        # s4 = s4.unsqueeze(0)
        # s5 = s5.unsqueeze(0)
        # s6 = s6.unsqueeze(0)

        s_pre = torch.cat((s1, s2, s3, s4, s5, s6, s7), dim=1)
        s = self.fc1(s_pre)
        s = self.activate_func(s)
        s = self.fc2(s)
        s = self.activate_func(s)
        return torch.softmax(self.fc3(s), dim=1)

    def do(self, s):
        with torch.no_grad():
            s1 = s[:, 0:8]
            # print(s1)
            s1 = self.pre1_0(s1)
            s1 = self.activate_func(s1)
            s1 = self.pre1_1(s1)
            s1 = self.activate_func(s1)

            s2 = s[:, 8:16]
            s2 = self.pre2_0(s2)
            s2 = self.activate_func(s2)
            s2 = self.pre2_1(s2)
            s2 = self.activate_func(s2)

            s3 = s[:, 16:28]
            s3 = self.pre3_0(s3)
            s3 = self.activate_func(s3)
            s3 = self.pre3_1(s3)
            s3 = self.activate_func(s3)

            s4 = s[:, 28:33]
            s4 = self.pre4_0(s4)
            s4 = self.activate_func(s4)

            s5 = s[:, 33:42]
            s5 = self.pre5_0(s5)
            s5 = self.activate_func(s5)
            s5 = self.pre5_1(s5)
            s5 = self.activate_func(s5)

            s6 = s[:, 42:56]
            s6 = self.pre6_0(s6)
            s6 = self.activate_func(s6)
            s6 = self.pre6_1(s6)
            s6 = self.activate_func(s6)

            s7 = s[:, 56:66]
            s7 = self.pre7_0(s7)
            s7 = self.activate_func(s7)

            # s1 = s1.unsqueeze(0)
            # s2 = s2.unsqueeze(0)
            # s3 = s3.unsqueeze(0)
            # s4 = s4.unsqueeze(0)
            # s5 = s5.unsqueeze(0)
            # s6 = s6.unsqueeze(0)

            s_pre = torch.cat((s1, s2, s3, s4, s5, s6, s7), dim=1)
            s = self.fc1(s_pre)
            s = self.activate_func(s)
            s = self.fc2(s)
            s = self.activate_func(s)
            return torch.softmax(self.fc3(s), dim=1)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        # self.hidden_state = self.init_hidden_state()
        self.pre1_0 = nn.Linear(8, 128)
        self.pre1_1 = nn.Linear(128, 256)  # 256
        self.pre2_0 = nn.Linear(8, 128)
        self.pre2_1 = nn.Linear(128, 256)  # 256
        self.pre3_0 = nn.Linear(12, 128)
        self.pre3_1 = nn.Linear(128, 256)  # 256
        self.pre4_0 = nn.Linear(5, 32)  # 32
        self.pre5_0 = nn.Linear(9, 128)
        self.pre5_1 = nn.Linear(128, 256)  # 256
        self.pre6_0 = nn.Linear(14, 128)
        self.pre6_1 = nn.Linear(128, 256)  # 256b
        self.pre7_0 = nn.Linear(10, 128)

        self.fc1 = nn.Linear(1312 + 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.GRUnet = CriticGRUNetwork(512, 256)

        # self.GRUnet= SimpleGRUNetwork(512,512)

        self.activate_func = [nn.ReLU(), nn.Tanh()][1]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.pre1_0)
            orthogonal_init(self.pre1_1)
            orthogonal_init(self.pre2_0)
            orthogonal_init(self.pre2_1)
            orthogonal_init(self.pre3_0)
            orthogonal_init(self.pre3_1)
            orthogonal_init(self.pre4_0)
            orthogonal_init(self.pre5_0)
            orthogonal_init(self.pre5_1)
            orthogonal_init(self.pre6_0)
            orthogonal_init(self.pre6_1)
            orthogonal_init(self.fc3)

        # print(self.hidden_state.shape)

    def init_hidden_state(self, length):
        return torch.zeros(length, 1, 2048)

    def forward(self, s):
        s1 = s[:, 0:8]
        # print(s1)
        s1 = self.pre1_0(s1)
        s1 = self.activate_func(s1)
        s1 = self.pre1_1(s1)
        s1 = self.activate_func(s1)
        s2 = s[:, 8:16]
        s2 = self.pre2_0(s2)
        s2 = self.activate_func(s2)
        s2 = self.pre2_1(s2)
        s2 = self.activate_func(s2)

        s3 = s[:, 16:28]
        s3 = self.pre3_0(s3)
        s3 = self.activate_func(s3)
        s3 = self.pre3_1(s3)
        s3 = self.activate_func(s3)

        s4 = s[:, 28:33]
        s4 = self.pre4_0(s4)
        s4 = self.activate_func(s4)

        s5 = s[:, 33:42]
        s5 = self.pre5_0(s5)
        s5 = self.activate_func(s5)
        s5 = self.pre5_1(s5)
        s5 = self.activate_func(s5)

        s6 = s[:, 42:56]
        s6 = self.pre6_0(s6)
        s6 = self.activate_func(s6)
        s6 = self.pre6_1(s6)
        s6 = self.activate_func(s6)

        s7 = s[:, 56:66]
        s7 = self.pre7_0(s7)
        s7 = self.activate_func(s7)

        s_pre = torch.cat((s1, s2, s3, s4, s5, s6, s7), dim=1)

        s = self.fc1(s_pre)
        s = self.activate_func(s)
        s = self.fc2(s)
        s = self.activate_func(s)
        return self.fc3(s)

    def cal(self, s):
        with torch.no_grad():
            s1 = s[:, 0:8]
            # print(s1)
            s1 = self.pre1_0(s1)
            s1 = self.activate_func(s1)
            s1 = self.pre1_1(s1)
            s1 = self.activate_func(s1)
            s2 = s[:, 8:16]
            s2 = self.pre2_0(s2)
            s2 = self.activate_func(s2)
            s2 = self.pre2_1(s2)
            s2 = self.activate_func(s2)

            s3 = s[:, 16:28]
            s3 = self.pre3_0(s3)
            s3 = self.activate_func(s3)
            s3 = self.pre3_1(s3)
            s3 = self.activate_func(s3)

            s4 = s[:, 28:33]
            s4 = self.pre4_0(s4)
            s4 = self.activate_func(s4)

            s5 = s[:, 33:42]
            s5 = self.pre5_0(s5)
            s5 = self.activate_func(s5)
            s5 = self.pre5_1(s5)
            s5 = self.activate_func(s5)

            s6 = s[:, 42:56]
            s6 = self.pre6_0(s6)
            s6 = self.activate_func(s6)
            s6 = self.pre6_1(s6)
            s6 = self.activate_func(s6)

            s7 = s[:, 56:66]
            s7 = self.pre7_0(s7)
            s7 = self.activate_func(s7)

            s_pre = torch.cat((s1, s2, s3, s4, s5, s6, s7), dim=1)

            s = self.fc1(s_pre)
            s = self.activate_func(s)
            s = self.fc2(s)
            s = self.activate_func(s)
            return self.fc3(s)


class PPO_discrete:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args)
        self.critic = Critic(args)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5
            )
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c, eps=1e-5
            )
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a
            )
            self.optimizer_critic = torch.optim.Adam(
                self.critic.parameters(), lr=self.lr_c
            )

    def evaluate(
        self, s
    ):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        return np.argmax(a_prob)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            probs = self.actor.do(s)
            # print(probs)
            dist = Categorical(probs)
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self, replay_buffer, total_steps):
        (
            s,
            a,
            a_logprob,
            r,
            s_,
            dw,
            done,
        ) = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient  #不需要梯度的地方
            # print(s)
            # self.critic.GRUnet.reset_hidden_state(1)    #critic初始化，算一系列的初态
            vs = self.critic.cal(s)
            # self.critic.GRUnet.reset_hidden_state(1)    #critic初始化，算一系列次态
            vs_ = self.critic.cal(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(
                reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())
            ):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                # self.critic.GRUnet.reset_hidden_state(1)    #actor和critic同时初始化，后面都得用到
                # self.actor.GRUnet.reset_hidden_state(1)
                # for index in range(self.batch_size):
                # if True:
                # index = list(range(501))
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(
                    -1, 1
                )  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(
                    -1, 1
                )  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(
                    a_logprob_now - a_logprob[index]
                )  # shape(mini_batch_size X 1)

                surr1 = (
                    ratios * adv[index]
                )  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                )  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic.forward(s[index])  # 这里才真的要梯度
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                # with torch.no_grad():
                #     print(critic_loss)
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_c_now
