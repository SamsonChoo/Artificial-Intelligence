import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import copy
from torch.distributions.categorical import Categorical

env = gym.make('CartPole-v0').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device( "cpu")

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

#overwrite the DQN
class DQN(nn.Module):

    def __init__(self, inputs, outputs): 
        super(DQN, self).__init__()
        intermediate_nodes = 128
        #1 hidden layer
        self.fc1 = nn.Linear(inputs, intermediate_nodes)
        self.fc_action = nn.Linear(intermediate_nodes, outputs)
        self.fc_value = nn.Linear(intermediate_nodes, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        # replaced relu with leaky_relu
        x = F.leaky_relu(self.fc1(x))
        action = F.softmax(self.fc_action(x), dim= 1)
        value = self.fc_value(x)
        
        return action, value


# BATCH_SIZE = 128
GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.02
# EPS_DECAY = 200
EPS = 0.25
# TARGET_UPDATE = 30
# EPS_END_STEPS = 12000
# REPLAY_MEMORY_SIZE = 50000
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.000005 #how much is mild?


# Get number of actions from gym action space
n_actions = env.action_space.n
n_states = env.observation_space.shape
policy_net = DQN(n_states[0], n_actions).to(device)
target_net = DQN(n_states[0], n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
# memory = ReplayMemory(REPLAY_MEMORY_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
#     sample = random.random()
    eps_threshold = EPS
    steps_done += 1
    
    action_dist, value_dist = policy_net(state)
    action_dist = action_dist*(1-eps_threshold) + eps_threshold/n_actions
    action = Categorical(action_dist).sample()
    return action, Categorical(action_dist).log_prob(action), value_dist      
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        global max_averaged_reward
        max_averaged_reward = max(means)
        
        threshold = np.array([195]*len(durations_t))
        plt.plot(threshold)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))

#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.uint8)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                                 if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)

#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch) #taken from the policy net
    
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.  
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch #expected from the target    
#     # Compute Huber loss
#     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#     #Compute MSE loss
#     #loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

def optimize_model(state, reward, logprobs, value):
    
    R = 0  #initial R
    qsa = []
    for r in reward[::-1]:
        R = r + R*GAMMA
        qsa.insert(0, R) 
    qsa = torch.tensor(qsa)
    
    #qsa baseline
    value = torch.tensor(value)
    value_detached = value.detach()
    qsa = qsa - value_detached
    
    loss = []
    for i in range(len(logprobs)):
        loss.append(qsa[i] * -1.0*logprobs[i])
    value_loss = F.smooth_l1_loss(value, qsa.detach())
    optimizer.zero_grad()
    loss = torch.cat(loss).sum() + value_loss
    loss.backward()
    optimizer.step()

max_averaged_reward = 0
num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state. and obtain the observations
    state = env.reset()
    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
    
    state_list = []
    reward_list = []
    logprobs_list = []
    value_list = []
    
    for t in count():
        # Select and perform an action.
        action, logprob, value = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, dtype = torch.float32, device=device).unsqueeze(0)
        if t > 500:
            done = True     
        if done:
            next_state = None
        reward = torch.tensor([reward], device=device)            
#         memory.push(state, action, next_state, reward)

        state_list.append(state)
        reward_list.append(reward)
        logprobs_list.append(logprob)
        value_list.append(value)
        
        state = copy.deepcopy(next_state)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
            
    optimize_model(state_list, reward_list, logprobs_list, value_list)
            
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

print("Max averaged reward: " + str(max_averaged_reward))

env.close()
plt.ioff()
plt.show()