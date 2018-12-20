import os
from collections import deque, namedtuple
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from expert.basic_expert import BaseAbility, Action
from trading import TradeAction

PATH = f"{os.path.dirname(__file__)}/model.ph"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'
consider_steps = 3
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def forget(self, rate=random.uniform(0.6, 0.9)):
        forget_rate = rate * len(self.memory)
        for i in range(int(forget_rate)):
            self.memory.pop()


class TradeOnError(nn.Module):

    def __init__(self):
        super(TradeOnError, self).__init__()
        self.lstm_1 = nn.LSTM(consider_steps, 20, 4)
        self.activation = nn.Softmax()
        self.out = nn.Linear(20, 3)

    def forward(self, x):
        x, h1 = self.lstm_1(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.activation(x)
        x = self.out(x)
        x = x.view(s, b, -1)
        return x


class TrailOnErrorDecision(BaseAbility):
    learn_rate = 0.9
    learn_decay = 0.995

    max_memory_limit = 100
    action_randomness = 1
    randomness_decay = 0.9995

    pre_status = deque(maxlen=consider_steps)
    action_taken_state = None
    last_action_raw = None
    batch_size = 72

    GAMMA = 0.999

    tot_rewards = 0
    tot_pos_rewards = 0
    tot_iterations = 0

    def __init__(self):
        self.memory = ReplayMemory(self.max_memory_limit)

        if os.path.exists(PATH):
            self.brain = torch.load(PATH)
            self.brain.eval()
            self.brain.train()
        else:
            self.brain = TradeOnError().to(device)

        self.target_net = TradeOnError().to(device)
        self.target_net.load_state_dict(self.brain.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.brain.parameters())
        self.criterion = nn.MSELoss()

    def get_state(self, states):
        for state in states:
            self.pre_status.append(state[1:2][0])

        if len(self.pre_status) == self.pre_status.maxlen:
            state = np.array(self.pre_status)
            state = state.reshape(-1, 1, consider_steps)
            return state
        # time.sleep(0.1)
        return None

    def act(self, state, result) -> Action:
        state = self.get_state(state)

        if result is not None:
            self.evaluate(state, result)

        action = None

        if state is not None:

            self.action_taken_state = state

            decision_val = random.uniform(0, 1)

            if decision_val < self.action_randomness:
                action_result = np.random.rand(3, 1)
                action_result = np.reshape(action_result, action_result.shape[0], None)
                # print(f"ran {action_result}")
            else:
                state = torch.from_numpy(state).to(device).float()
                action_result = self.brain(state).cpu().detach().numpy()
                action_result = np.reshape(action_result, action_result.shape[2], None)
                # print(f"brain {action_result}")

            self.last_action_raw = action_result

            certainty = np.max(action_result)
            action_index = np.unravel_index(action_result.argmax(), action_result.shape)

            action = Action.action_gen(TradeAction.STAY, certainty)
            if action_index == 0:
                action = Action.action_gen(TradeAction.SELL, certainty)
            elif action_index == 3:
                action = Action.action_gen(TradeAction.BUY, certainty)

            self.last_action = action

        return action

    def evaluate(self, state, actual_action):

        next_state = state
        pre_state = self.action_taken_state

        if next_state is not None and pre_state is not None:
            reward = self.reward(self.last_action, actual_action)

            self.tot_rewards += reward
            self.tot_pos_rewards += reward if reward > 0 else 0
            self.tot_iterations += 1

            self.memory.push(
                pre_state, self.last_action_raw, next_state, reward
            )

        if len(self.memory) >= self.max_memory_limit:
            self.train()
            self.action_randomness = self.action_randomness * self.randomness_decay
            self.memory.forget()
            self.save()

            print(f"Memory left {len(self.memory)}")
            time.sleep(0.5)

    def reward(self, predict_action, actual_action):

        if actual_action.type == TradeAction.BUY and predict_action.type == TradeAction.SELL:
            reward = (1 + actual_action.certainty) - predict_action.certainty
        elif actual_action.type == TradeAction.SELL and predict_action.type == TradeAction.BUY:
            reward = actual_action.certainty - (1 + predict_action.certainty)
        else:
            reward = actual_action.certainty - predict_action.certainty

        reward = 1 - abs(reward)

        #print(f"{reward} {predict_action, actual_action}")
        # time.sleep(0.1)
        return reward

    def train(self):
        BATCH_SIZE = self.batch_size
        GAMMA = self.GAMMA

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # print(len(transitions), len(self.memory))
        # print(transitions)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        next_states = np.array(batch.next_state)
        next_states = np.reshape(next_states,
                                 (next_states.shape[0], 1,
                                  next_states.shape[3]))
        next_states = torch.from_numpy(next_states).to(device).float()
        # print(non_final_next_states)

        state_batch = np.array(batch.state)
        state_batch = np.reshape(state_batch, (state_batch.shape[0], 1, state_batch.shape[3]))
        state_batch = torch.from_numpy(state_batch).to(device).float()
        # print(batch.action)
        action_batch = np.array(batch.action)
        # print(action_batch.shape)
        action_batch = torch.from_numpy(action_batch).to(device).float()
        reward_batch = torch.from_numpy(np.array(batch.reward)).to(device).float()

        output_1_batch = self.brain(next_states)
        reward_batch = np.reshape(reward_batch.cpu(), (reward_batch.shape[0], 1, 1))
        # print(reward_batch)
        # print(reward_batch.shape)
        # print(output_1_batch.shape)
        # print(GAMMA + torch.max(output_1_batch))
        # print()

        y_batch = reward_batch + GAMMA * output_1_batch.cpu()
        # torch.cat(tuple(reward_batch + GAMMA * torch.max(output_1_batch)))
        q_value = torch.sum(self.brain(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()

        y_batch = y_batch.detach()

        # print(q_value.shape)
        # print(y_batch.shape)
        # calculate loss
        loss = self.criterion(q_value, y_batch.to(device))

        # do backward pass
        loss.backward()
        self.optimizer.step()

        self.learn_rate = self.learn_rate * self.learn_decay

    def status(self):
        print("--Trail On Error Ability --")
        print(f"Left Memory {len(self.memory)}")
        print(f"Learn Rate {self.learn_rate}")
        print(f"Randomness  {self.action_randomness}")
        if self.tot_rewards != 0:
            print(f"iterations {self.tot_iterations}")
            print(f"Rewards  {self.tot_pos_rewards}/{self.tot_rewards} = ")

    def save(self):
        torch.save(self.brain, PATH)
