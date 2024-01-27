import copy
import torch
import pickle
import itertools
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from itertools import count
from collections import defaultdict
from torch.distributions.categorical import Categorical

device = 'cuda:5'

floattensor = lambda x: torch.FloatTensor(np.array(x)).to(device)
longtensor = lambda x: torch.LongTensor(np.array(x)).to(device)

class Reward_Function:
    def __init__(self, H, r0=1e-1, r1=0.5, r2=2.0):
        self.H = H
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2

    def compute(self, state):
        # e.g. state = [2,3]
        abs_val = abs(state/self.H - ((self.H-1)/self.H)/2)
        return self.r0 + self.r1*(abs_val > 0.25).prod(-1) + self.r2*((0.3 < abs_val) * (abs_val < 0.4)).prod(-1)

class GridEnv:
    def __init__(self, H, ndim, reward_fn, id):
        self.H = H
        self.ndim = ndim
        self.action_space_len = ndim # each coordinate
        self.reward_fn = reward_fn
        self._true_density = None
        self._state = None
        self.id = id

    def observe(self, state=None):
        state = np.int32(self._state if state is None else state)
        one_hot = np.zeros((self.H * self.ndim), dtype=np.float32) # flatten one-hot vector coordinate
        one_hot[np.arange(self.ndim)*self.H + state] = 1
        '''
        for example, let's say ndim=2, H=8, and state = [2,3]
        then,  one_hot = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0] (the first 8 slots for the first dimension and so on)
        selected index is [0,1]*8+[2,3] = [2, 11]
        making one_hot = [0,0,1,0,0,0,0,0, 0,0,0,1,0,0,0,0] 
        which corresponds to [2,3]
        '''
        return one_hot

    def reset(self):
        self._state = np.int32([0] * self.ndim) #[0,0] by defalut
        # reward = self.reward_fn(self.state_to_coord(self._state))
        reward = self.reward_fn(self._state)

        return self.observe(), reward, self._state 
        # e.g. [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0], reward, [0,0]

    def step(self, action, state_=None):
        # action: scalar -> inidicating which axis to move
        state = copy.deepcopy(self._state if state_ is None else state_)

        if action < self.action_space_len: 
            state[action] += 1
            done = state.max() >= self.H-1
        elif action == self.action_space_len: # stop action
            done = True
        
        if state_ is None:
            self._state = state

        reward = 0 if not done else self.reward_fn(state)
        return self.observe(state), reward, done, state
        # e.g. [0,0,1,0,0,0,0,0, 0,0,0,1,0,0,0,0], reward, T/F, [2,3]

    def find_previous_transitions(self, state, is_stop_action_used):
        if is_stop_action_used:
            return [self.observe(state)], [self.action_space_len] # [the same state], [stop_action]

        parent_states = []
        actions = []

        for action in range(self.action_space_len):
            if state[action] > 0:
                parent_state = copy.deepcopy(state)
                parent_state[action] -= 1
                if parent_state.max() == self.H - 1: # terminal parent -> ignore
                    # e.g. in 8x8 grid world, if the current state is [3,7], this algorithm also investigate [2,7], which is terminal state, too.
                    # this case should be ignored and the other parent candidate state [3,6] is the only parent state
                    continue
                parent_states += [self.observe(parent_state)]
                actions += [action]
        
        return parent_states, actions # one-hot parent states
    
    def true_density(self):
        if self._true_density != None:
            return self._true_density
        all_states = np.int32(list(itertools.product(*[list(range(self.H))]*self.ndim)))
        masks = np.array([len(self.find_previous_transitions(state, False)[0]) > 0 or sum(state) == 0 for state in all_states]) # maks out if it has no parents or is not the start point
        rewards = self.reward_fn(all_states)[masks]
        
        self._true_density = (rewards / rewards.sum(), rewards, list(map(tuple, all_states[masks]))) # probability mass function (because it's discrete), logits, states
        
        return self._true_density

class ReplayBuffer:
    def __init__(self, buffer_size, env):
        self.buffer = []
        self.buffer_size = buffer_size
        self.env = env
    
    def add(self, state, reward):
        if len(self.buffer) < self.buffer_size or reward > self.buffer[0][0]: 
            # when there is a free space or 
            # new observation has higher reward than the minimum-reward-obs in the buffer
            self.buffer = sorted(self.buffer + [(reward, state)])[-self.buffer_size:]

    def generate_backward(self, reward, state):
        state = np.int8(state)
        is_stop_action_used = state.max() < self.env.H - 1
        done = True

        trajectory = []
        while state.sum() > 0: # not on the start point
            parent_states, actions = self.env.find_previous_transitions(state, is_stop_action_used)
            # add the transition
            trajectory.append([floattensor(transition) for transition in (parent_states, actions, [reward], [self.env.observe(state)], [done])])
            # Then randomly choose a parent state
            if not is_stop_action_used:
                rand_idx = np.random.randint(0, len(parent_states))
                action = actions[rand_idx]
                state[action] -= 1
            # intermediary states
            is_stop_action_used = False
            done = False
            reward = 0

        '''
        trajectory =
        [
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            ...
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]]
        ]
        '''
        return trajectory
    
    def sample(self, size):
        if len(self.buffer) == 0: # empty buffer
            return []
        
        indices = np.random.randint(0, len(self.buffer), size)
        sample = sum([self.generate_backward(*self.buffer[i]) for i in indices], [])
        '''
        [
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            ...
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],

            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            ...
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            
            ...
            
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]],
            ...
            [[parent1, parent2, ...], [action1, action2, ...], [reward], [one-hot current state], [done]]
        ]
        '''
        return sample


def make_mlp(layer_dims, act=nn.LeakyReLU(), tail=[]):
    '''return an MLP with no top layer activation'''
    return nn.Sequential(*(sum(
        [[nn.Linear(in_dim, out_dim)] + ([act] if n < len(layer_dims)-2 else [])
         for n, (in_dim, out_dim) in enumerate(zip(layer_dims, layer_dims[1:]))], []) + tail
    ))

class GFlowNetAgent:
    def __init__(self, H, ndim, hidden_dim, n_hidden, envs, replay_buffer_size, device):
        self.model = make_mlp([H * ndim] + [hidden_dim]*n_hidden + [ndim + 1])
        # layers: (H x ndim) -> hidden_dim -> ... -> hidden_dim -> (ndim + 1) (ndim for each axis and 1 for stop-action)
        # takes state and outputs action
        self.device = device
        self.model.to(device)
        self.envs = envs
        self.ndim = ndim
        self.action_space_len = ndim # each coordinate
        self.replaybuffer = ReplayBuffer(replay_buffer_size, envs[0])
    
    def sample(self, replay_sample_size, batch_size, state_visit_log):
        batch = []
        batch += self.replaybuffer.sample(size=replay_sample_size)
        states = floattensor([env.reset()[0] for env in self.envs]) # one-hot states: [batch_size x one-hot-dim]
        dones = [False] * batch_size

        while not all(dones):
            with torch.no_grad():
                actions = Categorical(logits=self.model(states)).sample() # [batch_dim x action_dim]

            steps = [env.step(action) for (env, action) in zip([env for (env, done) in zip(self.envs, dones) if not done], actions)] # one-hot state, reward, done, state
            parent_states_and_actions = [env.find_previous_transitions(new_state, action==self.action_space_len) 
                                         for env, action, (_, _, _, new_state) 
                                         in zip([env for (env, done) in zip(self.envs, dones) if not done], actions, steps)] # zip(not_done_envs, actions, steps)
            batch += [[floattensor(transition) for transition 
                       in (parents, actions, [reward], [onehot_state], [done])] 
                       for (parents, actions), (onehot_state, reward, done, state) 
                       in zip(parent_states_and_actions, steps)]
            
            c = count(0)
            not_done_indices = {i:next(c) for i in range(batch_size) if not dones[i]} # find the indices with done=False
            dones = [bool(steps[not_done_indices[i]][2] if not done else done) for i, done in enumerate(dones)] # update dones when steps[.][2] = True (indicating done or not)
            states = floattensor([step[0] for step in steps if not step[2]])

            for (_, reward, done, state) in steps:
                if done:
                    state_visit_log.append(tuple(state))
                    self.replaybuffer.add(tuple(state), reward)

        return batch

    def calculate_loss(self, data, verbose=False):
        parent_states, actions, rewards, states, dones = map(torch.cat, zip(*data))
        parents_indices = longtensor(sum([[i]*len(parent_states) for i, (parent_states, _, _, _, _) in enumerate(data)], []))
        
        F_theta_parents = self.model(parent_states)[torch.arange(parent_states.shape[0]), actions.long()]
        in_flow = torch.log(torch.zeros((states.shape[0],)).to(self.device).index_add_(0, parents_indices, torch.exp(F_theta_parents)))

        F_theta_current = self.model(states)
        F_theta_current = F_theta_current*(1-dones).unsqueeze(1) + dones.unsqueeze(1)*(-floattensor([1000]))
        out_flow = torch.logsumexp(torch.cat([torch.log(rewards)[:, None], F_theta_current], 1), 1)

        loss = torch.mean((in_flow - out_flow)**2)

        if verbose:
            # seperate the loss in two components
            with torch.no_grad():
                leaf_loss = torch.sum(((in_flow - out_flow) * dones)**2) / (torch.sum(dones) + 1e-20)
                inner_flow_loss = torch.sum(((in_flow - out_flow) * (1-dones))**2) / (torch.sum(1-dones) + 1e-20)
            return loss, leaf_loss, inner_flow_loss
        
        return loss

def calculate_empirical_loss(env, state_visit_log):

    # true probability mass function
    true_pmf, logits, states_coord = env.true_density() #  true_probability_mass_function, rewards, and states_coordinate
    true_pmf = floattensor(true_pmf)

    # estimated probability mass function
    histogram = defaultdict(int)
    for state_visited in state_visit_log:
        histogram[state_visited] += 1
    z = sum([histogram[coord] for coord in states_coord])
    estimated_pmf = floattensor([histogram[coord] / z for coord in states_coord])

    # Loss
    l1_loss = abs(estimated_pmf - true_pmf).mean().item()
    kl_divergence_loss = (true_pmf * torch.log(true_pmf / estimated_pmf)).sum().item()

    return l1_loss, kl_divergence_loss



# Environment Hyperparameters
H = 8 # horizon
ndim = 2 # 2-D grid

# Agent Hyperparameters
hidden_dim = 256
n_hidden = 2 # number of hidden layers
# Replay Buffer Hyperparameters
replay_buffer_size = 100
# Optimizer Hyperparameters
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.999

# Training Hyperparameters
train_steps = 10000
batch_size = 16
replay_sample_size = 2

reward_fn = Reward_Function(H).compute
env = GridEnv(H, ndim, reward_fn, id=1127)
envs = [GridEnv(H, ndim, reward_fn, id) for id in range(batch_size)]
agent = GFlowNetAgent(H, ndim, hidden_dim, n_hidden, envs, replay_buffer_size, device)

optimizer = torch.optim.Adam(agent.model.parameters(), learning_rate, betas=(beta1, beta2))

# metrics
losses = []
state_visit_log = []
empirical_losses = []

sampling_num = 1
calculate_empirical_loss_period = 10
num_sample_for_empirical_loss = 200000
log_period = 10

for i in range(train_steps):
    # sample data
    data = agent.sample(replay_sample_size, batch_size, state_visit_log)
    # calculate loss
    loss, leaf_loss, inner_flow_loss = agent.calculate_loss(data, verbose=True)
    # update
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append([loss.item(), leaf_loss.item(), inner_flow_loss.item()])
    
    # calculate empirical l1 loss
    if i % calculate_empirical_loss_period == 0:
        l1_loss, kl_div_loss = calculate_empirical_loss(env, state_visit_log[-num_sample_for_empirical_loss:])
        empirical_losses.append([l1_loss, kl_div_loss])

    # print out results during training
    if i % log_period == 0:
        if len(losses) != 0:
            loss_mean = np.array(list(map(lambda x:x[0], losses))[-100:]).mean()
            leaf_loss_mean = np.array(list(map(lambda x:x[1], losses))[-100:]).mean()
            inner_flow_loss_mean = np.array(list(map(lambda x:x[2], losses))[-100:]).mean()
            print('Loss:', loss_mean, 'Leaf Loss:', leaf_loss_mean, 'Inner Flow Loss:', inner_flow_loss_mean)
        if len(empirical_losses) != 0:
            l1_loss, kl_div_loss = empirical_losses[-1]
            print('Empirical L1 Loss:', l1_loss, 'KL Divergence:', kl_div_loss, end='\n\n')

with open('./experiment_log.pickle', 'wb') as file:
    pickle.dump(
        {
            'losses': np.float32(losses),
            'state_visit_log': np.int8(state_visit_log),
            'empirical_losses': empirical_losses,
            'true_density': env.true_density()[0]
        },
        file
    )
