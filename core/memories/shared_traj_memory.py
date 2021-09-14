import numpy as np
import random
import torch
import torch.multiprocessing as mp

from core.memory import Memory

class Trajactory():
    def __init__(self, traj_state, traj_action, traj_reward):
        self.traj_state = traj_state
        self.traj_action = traj_action
        self.traj_reward = traj_reward
        self.traj_len = len(traj_action)


class SharedTrajMemory(Memory): # define a dict each value is a traj, traj is a class which include 3 elements: state action reward
    def __init__(self, args):
        super(SharedTrajMemory, self).__init__(args)
        # params for this memory
        # setup
        self.pos = mp.Value('l', 0)
        self.full = mp.Value('b', False)

        if self.tensortype == torch.FloatTensor:
            self.states = torch.zeros((self.memory_size, ) + tuple(self.state_shape), dtype=torch.float32)
        elif self.tensortype == torch.ByteTensor:
            self.states = torch.zeros((self.memory_size, ) + tuple(self.state_shape), dtype=torch.uint8)

        self.actions = torch.zeros( self.memory_size, self.action_shape)
        self.rewards = torch.zeros( self.memory_size, self.reward_shape)
        self.terminals = torch.zeros(self.memory_size, self.terminal_shape)

        self.states.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.terminals.share_memory_()


        self.memory_lock = mp.Lock()

        self.sample_type = 1 # 0 is sample-based 1 is traj-based

    @property
    def size(self):
        if self.full.value:
            return self.memory_size
        return self.pos.value

    def _feed(self, trajactory, priority=0.):
        traj_state, traj_action, traj_reward = trajactory

        traj_state = self.tensortype(traj_state)
        traj_action = torch.LongTensor(traj_action)
        traj_reward = torch.unsqueeze(torch.FloatTensor(traj_reward), dim=1)
        traj_len = traj_state.size()[0]
        traj_terminal = torch.zeros(traj_len, 1)
        traj_terminal[-1] = 1

        if self.pos.value + traj_len < self.memory_size:
            self.states[self.pos.value: self.pos.value + traj_len] = traj_state
            self.actions[self.pos.value: self.pos.value + traj_len - 1] = traj_action
            self.rewards[self.pos.value: self.pos.value + traj_len - 1] = traj_reward
            self.terminals[self.pos.value: self.pos.value + traj_len] = traj_terminal

            self.pos.value = self.pos.value + traj_len
        elif self.pos.value + traj_len == self.memory_size:
            self.states[self.pos.value: self.memory_size] = traj_state[0:self.memory_size - self.pos.value]
            self.actions[self.pos.value: self.memory_size - 1] = traj_action[0:self.memory_size - self.pos.value]
            self.rewards[self.pos.value: self.memory_size - 1] = traj_reward[0:self.memory_size - self.pos.value]
            self.terminals[self.pos.value: self.memory_size] = traj_terminal[0:self.memory_size - self.pos.value]
            self.full.value = True
            self.pos.value = 0
        else:
            self.states[self.pos.value: self.memory_size] = traj_state[0:self.memory_size-self.pos.value]
            self.actions[self.pos.value: self.memory_size] = traj_action[0:self.memory_size - self.pos.value]
            self.rewards[self.pos.value: self.memory_size] = traj_reward[0:self.memory_size - self.pos.value]
            self.terminals[self.pos.value: self.memory_size] = traj_terminal[0:self.memory_size - self.pos.value]
            temp_len = self.memory_size - self.pos.value

            self.full.value = True
            self.pos.value = 0

            self.states[self.pos.value: traj_len - temp_len] = traj_state[temp_len: traj_len]
            self.actions[self.pos.value: traj_len - temp_len - 1] = traj_action[temp_len: traj_len - 1]
            self.rewards[self.pos.value: traj_len - temp_len - 1] = traj_reward[temp_len: traj_len - 1]
            self.terminals[self.pos.value: traj_len - temp_len] = traj_terminal[temp_len: traj_len]

            self.pos.value = traj_len - temp_len


    def _sample(self, batch_size):

        # sample-based sampling
        if self.sample_type == 0:
            sample_len = 0
            upper_bound = self.memory_size if self.full.value else self.pos.value

            while sample_len <= 1:
                batch_inds = np.random.randint(0, upper_bound, size=batch_size).tolist()
                b_id = batch_inds[0]
                if self.terminals[b_id] == 1:
                    b_id -= 1
                for ter_id in range(b_id, upper_bound):
                    if self.terminals[ter_id] == 1:
                        break
                sample_len = len(self.states[b_id: ter_id])
            return (self.states[b_id: ter_id], self.actions[b_id: ter_id-1], self.rewards[b_id: ter_id-1])

        # traj-based sampling
        elif self.sample_type == 1:
            sample_len = 0
            upper_bound = self.memory_size if self.full.value else self.pos.value
            while sample_len <= 10:
                batch_inds = np.random.randint(0, upper_bound, size=batch_size).tolist()
                b_id = batch_inds[0]
                if self.terminals[b_id] == 1:
                    b_id -= 1
                star_id = 0
                ter_id = 0
                for star_id in range(b_id, 0, -1):
                    if self.terminals[star_id] == 1:
                        star_id += 1
                        break
                for ter_id in range(b_id, upper_bound):
                    if self.terminals[ter_id] == 1:
                        ter_id += 1
                        break
                sample_len = ter_id - star_id

            return (self.states[star_id: ter_id], self.actions[star_id: ter_id-1], self.rewards[star_id: ter_id-1])



    def feed(self, trajactory, priority=0.):
        with self.memory_lock:
            return self._feed(trajactory, priority)

    def sample(self, batch_size):
        with self.memory_lock:
            return self._sample(batch_size)

