import random
from collections import deque
from os import path

import numpy as np
import torch

import config as cfg
from nn import DeepNeuralNet


class TrainMario:
    def __init__(self, state_space, action_space, log_loc, learning_mode):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Initializing Mario to train with device = \'{self.device}\'')

        # initialize self. variables for data we need in other methods, 
        # otherwise don't use self. to save memory
        self.action_space  = action_space
        self.log_loc       = log_loc
        self.learning_mode = learning_mode

        # model is our nn that we update constantly
        self.model = DeepNeuralNet(state_space, self.action_space).to(torch.float32).to(device=self.device)
        # using ADAM and Huber loss
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

        # double deep q learning requires a second, identical net that we freeze the weights for
        if self.learning_mode == 'ddqn':
            # target_model is our nn that we update infrequently
            self.target_model = DeepNeuralNet(state_space, self.action_space).to(torch.float32).to(device=self.device)
            self.target_optimizer = torch.optim.Adam(self.target_model.parameters(), lr=cfg.learning_rate)
            for parameter in self.target_model.parameters():
                parameter.requires_grad = False

        # epsilon-greedy selection hyperparameters
        self.epsilon           = cfg.starting_epsilon # starting explore value
        self.total_steps_taken = 0                    # track the total number of steps taken to know when to update/save nns
       
        # remember the last 5_000 actions in self.memory (the replay buffer). Once filled, the DEQueue will remove the 'memories' of the oldest moves to
        # make room for new ones. Could have implemented with a list, but maxlen makes implementation with DEQueue easier. maxlen constrained by gpu memory.
        self.memory     = deque(maxlen=5_000)
        self.batch_size = cfg.batch_size  # number of samples we use in td_estimate and td_target


    # update self.target_model to contain self.model's current weights
    def update_target_model_weights(self):
        # this only applies to ddqn since dqn doesn't use a target_model
        if self.learning_mode == 'ddqn':
            if self.total_steps_taken % cfg.update_ddqn_weights_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())


    # save a .pt training weights file containing nn weights and epsilon for testing purposes
    def save_model_weights(self):
        if self.total_steps_taken % cfg.save_weights_frequency == 0:
            weights_file = f'{self.learning_mode}_weights_{int(self.total_steps_taken)}_steps.pt'
            save_loc = path.join(self.log_loc, weights_file)
            # save weights and epsilon value for testing current progress in test_mario
            torch.save(dict(model=self.model.state_dict(), epsilon=self.epsilon), save_loc)
            print(f'Weights file {weights_file} saved to {save_loc} at step {self.total_steps_taken}')


    # use memories from previous experiences to make decisions
    def sample_memories(self):
        # returns a 'self.batch_size' length list of items chosen from 'self.memory' without replacement
        sampled_memories = random.sample(self.memory, self.batch_size)
        # rearranges our sampled tensors from a list of 'self.batch_size' (state, action, reward, new_state, done) tensors to individual 
        # property tensors, i.e. tensor(state), tensor(action), tensor(reward), tensor(new_state), tensor(done), each with 'self.batch_size' values within.
        # we use this to efficiently pass the values to td_estimate and td_target, which do not need all 5 of these values
        state, action, reward, new_state, done = map(torch.stack, zip(*sampled_memories))
        # squeeze() action, reward, and done since they are single values that can be put into a 1D tensor, rather than their current 2D tensors
        return state, action.squeeze(), reward.squeeze(), new_state, done.squeeze()


    # computes 'self.batch_size' td estimate values
    def td_estimate(self, state, action):
        batch_indexes = np.array([_ for _ in range(self.batch_size)]) # an array of each index of the batches
        model_outputs = self.model(state)                             # model outputs of dimensions 'env.action_space.n' x 'self.batch_size'
        selected_td_estimates = model_outputs[batch_indexes, action]  # for each output tensor in model_outputs, choose the value of the chosen action - we get a 'self.batch_size' tensor of td_estimates
        return selected_td_estimates.to(torch.float32)


    # this decorator ensures that the result of every computation in this function will have 
    # requires_grad=False, even when the inputs have requires_grad=True
    # computes 'self.batch_size' td target values
    @torch.no_grad()
    def td_target(self, reward, new_state, done):
        new_q = self.model(new_state)
        best_action = torch.argmax(new_q, axis=1)
        batch_indexes = np.array([_ for _ in range(self.batch_size)])
        # figure out the best next q value from target model if using ddqn, or from model if using dqn
        if self.learning_mode == 'ddqn':
            model_outputs = self.target_model(new_state)
        else:
            model_outputs = self.model(new_state)
        max_future_reward = (1 - done) * model_outputs[batch_indexes, best_action].to(torch.float32) # (1-done) lets us return 0 as the max_future_reward if our episode is done
        return (reward + (cfg.gamma * max_future_reward))


    # standard model backprop based on the 'self.batch_size' td_estimates and td_targets
    def update_model(self, td_estimate, td_target):
        loss = self.loss(td_estimate, td_target)
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()


    # choose an action given a state using epsilon greedy selection
    def choose_action(self, state):
        # exploration
        if np.random.rand() < self.epsilon:
            new_action = np.random.randint(self.action_space)
        # exploitation
        else:
            # unsqueeze transforms state from the (4, 84, 84) 'gym.wrappers.frame_stack.LazyFrames' to our required [1, 4, 84, 84] tensor
            state      = torch.from_numpy(np.array(state)).to(torch.float32).to(device=self.device).unsqueeze(0)
            actions    = self.model(state).to(device=self.device)
            new_action = torch.argmax(actions, axis=1).item()

        # update epsilon after every step until epsilon reaches its min value
        self.epsilon = max(cfg.min_epsilon, self.epsilon * cfg.epsilon_decay)

        # update the total number of steps taken
        self.total_steps_taken += 1
        return new_action


    # commit previously performed moves to memory
    def store_in_memory(self, state, action: int, reward: float, new_state, done: bool):
        # convert 'done' bools to floats for td_target
        self.memory.append((torch.tensor(np.array(state), device=self.device).to(torch.float32),
                            torch.tensor([action], device=self.device),
                            torch.tensor([reward], device=self.device),
                            torch.tensor(np.array(new_state), device=self.device).to(torch.float32),
                            torch.tensor([done], device=self.device).to(torch.float32)))


    def learn(self):
        # these updates should be done regardless of if we are actually learning yet
        self.update_target_model_weights()
        self.save_model_weights()

        # if we haven't sampled enough steps, don't start learning
        if self.total_steps_taken < cfg.start_learning_after:
            return

        # only learn every 'cfg.learning_frequency' steps
        if self.total_steps_taken % cfg.learning_frequency != 0:
            return

        state, action, reward, new_state, done = self.sample_memories()
        td_estimate = self.td_estimate(state, action)
        td_target   = self.td_target(reward, new_state, done)

        self.update_model(td_estimate, td_target)
