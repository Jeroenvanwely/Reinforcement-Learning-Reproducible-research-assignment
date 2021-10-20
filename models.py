import numpy as np
import time
import random

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5,  extra_actions=False, add_stochasticity=False):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        extra_action: If set to True there are 8 possible actions an agent can take, if set to False it will be 4.
        add_stochasticity: If set to True stochasticity will be added, if set to False it won't be.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
        iteration_time_list which is an list containing the time it took to run each iteration.
    """
    
    stats = [] # Keeps track of useful statistics
    iteration_time_list = [] # Keeps track of iteration time

    for i_episode in range(num_episodes):
        i, R, d = 0, 0, False
        s = env.reset() # Initialize state
        a = policy.sample_action(s, extra_actions) # Sample action from epsilon-greedy policy
        while d is False: # Until S is terminal loop for each step in episode
            
            begin_sarsa_iteration = time.time() # Define start time of iteration

            s_, r, d, _ = env.step(a) # Take action a and observe new state s_ and reward r
            # Choose action a_ from state s_ using policy derived from Q using epsilon-greedy
            a_ = policy.sample_action(s_, extra_actions)
            Q[s, a] = Q[s, a] + alpha * (r + discount_factor * Q[s_, a_] - Q[s, a]) # Update step

            # Add stochasticity to select next state
            if add_stochasticity and np.random.random_sample() < 0.2:
                if extra_actions:
                    random_action = np.random.choice([0,1,2,3,4,5,6,7])
                else:
                    random_action = np.random.choice([0,1,2,3])
                s, _, _, _ = env.step(random_action)
                a = policy.sample_action(s)
            else:
                s, a = s_, a_ # Update current state s and action a

            i += 1 # Update step-count i
            R += r # Update total reward R

            iteration_time_list.append(time.time() - begin_sarsa_iteration) # Append iteration time to list

        stats.append((i, R))

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), iteration_time_list


def expected_sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, extra_actions=False, add_stochasticity=False):
    """
    Expected sarsa algorithm. Finds the optimal greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Probability of picking non-greedy action.
        extra_action: If set to True there are 8 possible actions an agent can take, if set to False it will be 4.
        add_stochasticity: If set to True stochasticity will be added, if set to False it won't be.

    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
        iteration_time_list which is an list containing the time it took to run each iteration.
    """
    stats = [] # Keeps track of useful statistics
    iteration_time_list = [] # Keeps track of iteration time

    if extra_actions:
        n_actions = 8
    else:
        n_actions = 4

    for i_episode in range(num_episodes):
        i, R, d = 0, 0, False    
        s = env.reset() # Initialize state
        while d == False:
            
            begin_exp_sarsa_iteration = time.time()

            a = policy.sample_action(s, extra_actions) # Choose action a_ from state s_ using policy derived from Q using epsilon-greedy
            s_, r, d, _ = env.step(a) # Take action a and observe new state s_ and reward r

            # Compute policy probabilities
            policy_probs = [epsilon/n_actions]*n_actions # Every action at least epsilon/number_actions prob
            max_actions = np.argwhere(Q[s_] == np.amax(Q[s_])).flatten().tolist() # Get all maximum actions
            for index in max_actions:
                policy_probs[index] += (1-epsilon)/len(max_actions) # Add minimum probability for max actions with (1-epsilon)/number of maxium actions

            Q[s, a] = Q[s, a] + alpha * (r + discount_factor *  np.sum(Q[s_]*policy_probs) - Q[s, a]) # Update step

            # Add stochasticity to select next state
            if add_stochasticity and np.random.random_sample() < 0.2:
                if extra_actions:
                    random_action = np.random.choice([0,1,2,3,4,5,6,7])
                else:
                    random_action = np.random.choice([0,1,2,3])
                s, _, _, _ = env.step(random_action)
                a = policy.sample_action(s)
            else:
                s = s_ # Update current state s

            i += 1 # Update step-count i
            R += r # Update total reward R

            iteration_time_list.append(time.time() - begin_exp_sarsa_iteration) # Append iteration time to list

        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), iteration_time_list