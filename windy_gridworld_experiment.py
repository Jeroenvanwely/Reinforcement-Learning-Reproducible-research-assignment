import numpy as np
import matplotlib.pyplot as plt

from windy_gridworld import WindyGridworldEnv
from models import sarsa, expected_sarsa
from policies import EpsilonGreedyPolicy

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def run_windy_gridworld(method='sarsa', num_episodes=1000, discount_factor=1.0, alpha=0.5, average_over_n=50, epsilon=0.1, extra_actions=False, add_stochasticity=False):

    # Initialize variables
    env = WindyGridworldEnv(extra_actions=extra_actions)
    Q = np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)

    # If method is SARSA
    if method == 'sarsa':
        Q, (episode_lenghts, episode_returns), iteration_time_list = sarsa(env, policy, Q, num_episodes=num_episodes, 
                                                                    discount_factor=discount_factor, 
                                                                    alpha=alpha, extra_actions=extra_actions, 
                                                                    add_stochasticity=add_stochasticity)

    # If method is expected SARSA
    if method =='exp_sarsa':
        Q, (episode_lenghts, episode_returns), iteration_time_list = expected_sarsa(env, policy, Q, num_episodes=num_episodes, 
                                                                                discount_factor=discount_factor, 
                                                                                alpha=alpha, epsilon=epsilon, 
                                                                                extra_actions=extra_actions, 
                                                                                add_stochasticity=add_stochasticity)

    # Smooth the episode lengths and values
    episode_lenghts = running_mean(episode_lenghts, average_over_n)
    episode_returns = running_mean(episode_returns, average_over_n)
    
    return episode_lenghts, episode_returns, iteration_time_list