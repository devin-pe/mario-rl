'''HYPERPARAMETER CONFIGURATIONS'''

'''train.py hyperparameters'''
# training_episodes = 50_000
training_episodes   = 100_000 # how many episodes to train on (may stop earlier if convergence reached)
record_stats        = 10      # how often we record rewards to TensorBoard

'''train_mario.py hyperparameters'''
# learning_rate  = 0.001
# learning_rate  = 0.0001
# learning_rate  = 0.00005
learning_rate    = 0.00001     # how fast we learn - this one seems to work good from trial and error
starting_epsilon = 1.0         # start training by exploring all the time
# epsilon_decay  = 0.9_999_999 # way too long to reach min_epsilon
# epsilon_decay  = 0.99_999    # way too quick to reach min_epsilon
epsilon_decay    = 0.999_999   # the rate at which epsilon decreases - good value
# min_epsilon    = 0.10        # can try less, might work better if enough exploring done before reaching min_epsilon
min_epsilon      = 0.05        # min explore value - always explore at least every 1/20 moves
# batch_size     = 16          # GPU can handle more
batch_size       = 32          # how many memories we sample from memory while learning
gamma            = 0.90        # how much we value future rewards

save_weights_frequency          = 200_000 # save pt file every 200_000 steps
# update_ddqn_weights_frequency = 10_000
update_ddqn_weights_frequency   = 5_000   # average steps ~300 on completions, so 5_000 good for update freq - don't go smaller, then it becomes too similar to dqn
# start_learning_after          = 100_000 # unnecessarily long 
start_learning_after            = 10_000  # sample this many steps before starting to update model(s)
# learning_frequency            = 4       # can probably learn more often
learning_frequency              = 2       # how often we learn after sampling 'start_learning_after' steps