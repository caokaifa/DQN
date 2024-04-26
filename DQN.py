import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import os
from torch.utils.tensorboard import SummaryWriter

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0",render_mode="human")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape
weights_path="/home/smart/reinforcement_learning/Deep-reinforcement-learning-with-pytorch/Char01_DQN/DQN_model/cart_pole.pth"

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net() #训练网络、目标网络
        if os.path.exists(weights_path):
            self.eval_net = torch.load(weights_path)
            self.target_net=self.eval_net
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2)) # 存储状态、动作、奖励、下一个状态，memory_capacitity为存储容量
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) #网络优化方法、学习率
        self.loss_func = nn.MSELoss() #损失函数
        self.writer = SummaryWriter()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            # action = torch.max(action_value, 1)[1].data.numpy()
            action = torch.max(action_value, 1)[1].item()
            # print("----88888-----=",action)
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        
        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.target_net, weights_path)
            print("model saved to",weights_path)
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])
    
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5 #化为【-0.5,0.5】
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2

    # print("-----reward_func=----=",r1, r2,env.x_threshold,env.theta_threshold_radians),
    return reward

def main():
    dqn = DQN() #初始化DQN网络
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
   
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state,info = env.reset()
        ep_reward = 0
        while True:
            env.render()
            if 0:
                action = dqn.choose_action(state)#训练,当训练到稳定性一定时，不要贪婪策略训练了，直接稳定模型训练
            else:
                state_ = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
                action_value = dqn.eval_net.forward(state_)
                action = torch.max(action_value, 1)[1].item()
                # print("------6666----=",action)
                # action = np.random.randint(0,NUM_ACTIONS)
            next_state, reward_env, done,truncated, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot) #使用状态自己重新计算奖励

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward
            dqn.writer.add_scalar("Reward_env", i,reward_env)
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                    dqn.writer.add_scalar("Reward", i,reward)
            if done:
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        

if __name__ == '__main__':
    main()
