from os import path

import numpy as np
import torch

from nn import DeepNeuralNet


# We use this class to test weights as they are created in TrainMario. Can be used by TAs to
# view Mario's improvements over time
class TestMario:
    def __init__(self, state_space, action_space, log_loc, weights):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Initializing Mario to test with device = \'{self.device}\'')

        self.action_space  = action_space

        # get pretrained model weights and epsilon
        load_path    = path.join(log_loc, weights)
        pt_file      = torch.load(load_path)
        self.epsilon = pt_file.get('epsilon')
        state_dict   = pt_file.get('model')

        self.model = DeepNeuralNet(state_space, self.action_space).to(torch.float32).to(device=self.device)
        print(f"Loading {load_path} with epsilon = {np.round(self.epsilon, 5)}")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # choose an action given a state using epsilon greedy selection
    def choose_action(self, state):
        # exploration
        if np.random.rand() < self.epsilon:
            new_action = np.random.randint(self.action_space)
        else:
        # unsqueeze transforms state from the (4, 84, 84) 'gym.wrappers.frame_stack.LazyFrames' to our required [1, 4, 84, 84] tensor
            state      = torch.from_numpy(np.array(state)).to(torch.float32).to(device=self.device).unsqueeze(0)
            actions    = self.model(state).to(device=self.device)
            new_action = torch.argmax(actions, axis=1).item()

        return new_action
