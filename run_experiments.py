from windy_gridworld_experiment import run_windy_gridworld
import matplotlib.pyplot as plt
from tqdm import tqdm as _tqdm
import time
import argparse
import numpy as np
import os

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def plot_results(avg_episode_lenghts_sarsa, avg_episode_returns_sarsa, avg_episode_lenghts_exp_sarsa, avg_episode_returns_exp_sarsa, num_episodes, average_over_n, epochs):
    # Calculate std dev of lengths and returns over multiple epochs
    std_episode_lenghts_sarsa = np.std(avg_episode_lenghts_sarsa, 0)
    std_episode_returns_sarsa = np.std(avg_episode_returns_sarsa, 0)
    std_episode_lenghts_exp_sarsa = np.std(avg_episode_lenghts_exp_sarsa, 0)
    std_episode_returns_exp_sarsa = np.std(avg_episode_returns_exp_sarsa, 0)

    # Calculate average lengths and returns over multiple epochs
    avg_episode_lenghts_sarsa = np.mean(avg_episode_lenghts_sarsa, 0)
    avg_episode_returns_sarsa = np.mean(avg_episode_returns_sarsa, 0)
    avg_episode_lenghts_exp_sarsa = np.mean(avg_episode_lenghts_exp_sarsa, 0)
    avg_episode_returns_exp_sarsa = np.mean(avg_episode_returns_exp_sarsa, 0)

    # Plot average returns over multiple epochs including standard deviaton
    plt.plot(avg_episode_returns_sarsa, label='Sarsa mean', color='green')
    plt.fill_between(range(num_episodes-average_over_n), avg_episode_returns_sarsa - std_episode_returns_sarsa, 
                    avg_episode_returns_sarsa + std_episode_returns_sarsa, color='green', alpha=0.3, label="Sarsa std")

    plt.plot(avg_episode_returns_exp_sarsa, label='Expected Sarsa mean', color='red')
    plt.fill_between(range(num_episodes-average_over_n), avg_episode_returns_exp_sarsa - std_episode_returns_exp_sarsa, 
                    avg_episode_returns_exp_sarsa + std_episode_returns_exp_sarsa, color='red', alpha=0.3, label="Expected Sarsa std")
    plt.xlabel('Episodes averaged over {} runs'.format(epochs))
    plt.ylabel('Reward')
    
    if config.extra_actions and config.add_stochasticity:
        plt.title('Performance of Sarsa and expected Sarsa in stochastic windy \n gridworld with increased action-space using epsilon={}'.format(config.epsilon))
    elif config.extra_actions:
        plt.title('Performance of Sarsa and expected Sarsa in windy gridworld \nwith increased action-space using epsilon={}'.format(config.epsilon))
    elif config.add_stochasticity:
        plt.title('Performance of Sarsa and expected Sarsa in \nstochastic windy gridworld using epsilon={}'.format(config.epsilon))
    else:
        plt.title('Performance of Sarsa and expected Sarsa in \ndetermistic windy gridworld using epsilon={}'.format(config.epsilon))
    plt.legend()
    plt.show()


def main():
    
    avg_episode_lenghts_sarsa, avg_episode_returns_sarsa = [], []
    avg_episode_lenghts_exp_sarsa, avg_episode_returns_exp_sarsa = [], []

    begin_sarsa = time.time()
    iter_times = []
    for i in tqdm(range(config.epochs)):
        episode_lenghts_sarsa, episode_returns_sarsa, iteration_time_list = run_windy_gridworld(method='sarsa', num_episodes=config.num_episodes, 
                                                                        discount_factor=config.discount_factor, alpha=config.alpha, 
                                                                        average_over_n=config.average_over_n, epsilon=config.epsilon, 
                                                                        extra_actions=config.extra_actions, add_stochasticity=config.add_stochasticity)
        avg_episode_lenghts_sarsa.append(episode_lenghts_sarsa)
        avg_episode_returns_sarsa.append(episode_returns_sarsa)
        iter_times += iteration_time_list
    print("Average iteration time for Sarsa: {}".format(np.mean(np.array(iter_times))))
    end_sarsa = time.time()
    time_sarsa = end_sarsa - begin_sarsa

    begin_exp_sarsa = time.time()
    iter_times = []
    for i in tqdm(range(config.epochs)):
        episode_lenghts_exp_sarsa, episode_returns_exp_sarsa, iteration_time_list = run_windy_gridworld(method='exp_sarsa', num_episodes=config.num_episodes, 
                                                                        discount_factor=config.discount_factor, alpha=config.alpha, 
                                                                        average_over_n=config.average_over_n, epsilon=config.epsilon,
                                                                        extra_actions=config.extra_actions, add_stochasticity=config.add_stochasticity)

        avg_episode_lenghts_exp_sarsa.append(episode_lenghts_exp_sarsa)
        avg_episode_returns_exp_sarsa.append(episode_returns_exp_sarsa)
        iter_times += iteration_time_list
    print("Average iteration time for expected Sarsa: {}".format(np.mean(np.array(iter_times))))
    end_exp_sarsa = time.time()
    time_exp_sarsa = end_exp_sarsa - begin_exp_sarsa

    print('Time Sarsa: ', time_sarsa)
    print('Time Expected Sarsa: ', time_exp_sarsa)

    plot_results(avg_episode_lenghts_sarsa, 
                avg_episode_returns_sarsa, 
                avg_episode_lenghts_exp_sarsa, 
                avg_episode_returns_exp_sarsa, 
                config.num_episodes, 
                config.average_over_n, 
                config.epochs)

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes per run.')
    parser.add_argument('--discount_factor', type=float, default=1.0,
                        help='Factor that controls how much we care about future actions.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Update step size.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Probability of picking random action.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of runs we want to average the plots over.')
    parser.add_argument('--average_over_n', type=int, default=50,
                        help='Smooth the graph by averaging over every n episodes.')

    parser.add_argument('--extra_actions', dest='extra_actions', action='store_true')
    parser.add_argument('--no_extra_actions', dest='extra_actions', action='store_false')
    parser.set_defaults(extra_actions=False)

    parser.add_argument('--add_stochasticity', dest='add_stochasticity', action='store_true')
    parser.add_argument('--no_add_stochasticity', dest='add_stochasticity', action='store_false')
    parser.set_defaults(add_stochasticity=False)

    config = parser.parse_args()
    main()
