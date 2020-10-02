import numpy as np
import random
import os
from torch import cat, Tensor, LongTensor
from torch.nn import Module, Linear
from torch.nn.functional import relu, softmax, smooth_l1_loss
from torch.optim import Adam
import torch.autograd as autograd
from torch.autograd import Variable
from queue import Queue

# Can change these to affect various parameters of the dqn
# Number of hidden layer neurons
NUM_HL_NEURONS = 30
# Max number of events being held in memory
MEMORY_CAPACITY = 100000
# Learning rate for optimizer
LEARNING_RATE = 0.001
# Temperature parameter for softmax function
TEMPERATURE = 7


class Network(Module):
    '''The neural network that runs brain'''

    def __init__(self, input_size, num_actions):
        '''Constructor for the network'''
        super(Network, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        # This is the input layer to hidden layer
        self.fc1 = Linear(input_size, NUM_HL_NEURONS)
        # This is the hidden layer to output action
        self.fc2 = Linear(NUM_HL_NEURONS, num_actions)

    def forward(self, state):
        '''Activates the neurons (forward propagation)'''
        # Pass state through first layer with a rectifier function
        hidden_neurons = relu(self.fc1(state))
        # Get q values from hidden layer to output
        q_values = self.fc2(hidden_neurons)
        return q_values


class ReplayMemory(object):
    '''Keeps track of given amount of previous events'''

    def __init__(self, capacity):
        '''Constructor for replay memory'''
        self.memory = Queue(capacity)

    def push(self, event):
        ''' Method for adding an event to memory, keeps memory stack under
            capacity
            event object has four elements:
                st - last state
                st+1 - next state
                at - action just played
                rt - reward for action
        '''
        if self.memory.full():
            self.memory.get()
        self.memory.put(event)

    def sample(self, batch_size):
        '''Method for sampling a specific number of events'''
        # zip(*list) turns list like [(1,2,3), (4,5,6)]
        # to [(1,4), (2,5), (3,6)]
        # need to do this for use with pytorch to get tensor and gradient
        samples = zip(*random.sample(self.memory, batch_size))

        # Lambda function takes samples and concatenates with respect to
        # state which is the first variable
        return map(lambda sample: Variable(cat(sample, 0)), samples)


class Dqn:
    '''Deep Q learning network powering self driving car'''

    def __init__(self, input_size, num_actions, gamma):
        ''' Constructor for our deep q network initialising neural net and
            replay memory
        '''
        self.gamma = gamma
        # sliding window of the mean of last 100 rewards evolving over time
        self.reward_window = []
        self.model = Network(input_size, num_actions)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        # Connect the params of our network to Adam optimizer
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        # State is 5d vector encoding 3 sensor inputs, orientation, and
        # -orientation which is created as a torch tensor
        # Creating a first 'fake' dimension for tensor with unsqueeze
        self.last_state = Tensor(input_size).unsqueeze(0)
        # Last action taken represented by an index
        self.last_action = 0
        # Last reward received between -1 and 1
        self.last_reward = 0

    def select_action(self, state):
        '''Method for car to make an action at each given time'''
        probabilities = softmax(
            # Turn state tensor into a variable without gradient
            self.model(Variable(state, volatile=True))*TEMPERATURE
        )
        action = probabilities.multinomial()
        # Action to take stored at index 0, 0
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        ''' Method for training AI, forward propagation and back propagation
            with gradient descent to adjust weights
        '''
        # Collect all actions taken, need to unsqueeze to compensate for fake
        # dimension, then squeeze the fake dimension after gathering
        outputs = self.model(
            batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # get the max Q values of all possible next states
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # targets is the max of Q values * gamma + reward
        target = self.gamma * next_outputs + batch_reward
        # Find temporal difference loss for backpropagation
        td_loss = smooth_l1_loss(outputs, target)
        # zero grad reinitializes optimizer at each step of gradient descent
        self.optimizer.zero_grad()
        # backpropagation of error in network
        td_loss.backward(retain_variables=True)
        # updates the weights in the network
        self.optimizer.step()

    def update(self, reward, new_signal):
        '''Method for updating the reward and state to get next action'''
        # convert signal list into a torch tensor with float elements and add
        # fake batch dimension
        new_state = Tensor(new_signal).float().unsqueeze(0)
        # must make sure all elements are torch tensors
        self.memory.push((
            self.last_state,
            new_state,
            # conversion of simple number into a torch tensor
            LongTensor([int(self.last_action)]),
            # tensor of simple float reward
            Tensor([self.last_reward])
        ))
