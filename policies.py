import numpy as np
import random

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs, extra_actions=False):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state
            extra_actions: If set to True choose out of 8 actions, 4 actions otherwise.

        Returns:
            An action (int).
        """
        
        # Define the actions
        if extra_actions: # 8 actions if extra_actions is True
            actions = [0,1,2,3,4,5,6,7]
        else: # 4 actions if extra_actions is False
            actions = [0,1,2,3]

        # With probabilty epsilon, pick random action
        random_digit = np.random.random_sample() # Get random digit
        if random_digit < self.epsilon: # If epsilon is smaller than the random digit, pick random action
            action = np.random.choice(actions)
        # If epsilon is bigger than the random digit, pick random greedy action.
        # Note that if there is only one greedy action this will always be selected
        else: # If epsilon is bigger than the random digit, pick random greedy action.
            action = np.random.choice(np.argwhere(self.Q[obs] == np.amax(self.Q[obs])).flatten())

        return action