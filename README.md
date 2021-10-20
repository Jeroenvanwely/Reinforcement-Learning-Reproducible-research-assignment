# Reinforcement-Learning-Reproducible-research-assignment

## Information & Authors

Authors:
Jochem Soons    -   11327030
Jeroen van Wely -   11289988
Niek IJzerman   -   11318740

This repository contains the code for Reproducible research assignment.
The assignment is part the course Reinfocement Learning course of the Master programme Artificial Intelligence at the University of Amsterdam.

## Files Included

- run_experiments.py
 - Contains the code to define and run the experiment and get the plots.

- windy_gridworld.py
 - Contains the code for the windy_gridworld class.

- models.py
 - Contains the code for the Sarsa algorithms.
 - Contains the code for the Expected Sarsa algorithm

- policies.py
 - Contains the code for the epsilon greedy policy class.

- windy_gridworld_experiment.py
 - Contains the code to run the windy grid world experiment, which was defined in run_experiment.py, on either Sarsa or Expected Sarsa. 

## Requirements

We used a conda environment for running our code, that we exported as yml file: see environment.yml.

To create the environment, run:

    conda env create -f environment.yml

To activate the environment, run:

    conda activate rlproject

## How to run the code

To run the code:
 - All files will have to be within the same directory.
 - To run an experiment run: "python3 run_experiments.py" Followed by command line argument specification.
 - An example would be: python3 run_experiments.py --num_episodes=1000 --discount_factor=1.0 --alpha=0.5 --epsilon=0.1 --epochs=10 --average_over_n=50 --no_extra_actions --no_add_stochasticity
 - To run the experiment using extra actions one would have to change "--no_extra_actions" to "--extra_actions". Likewise to add stochasticity change "--no_add_stochasticity" to "--add_stochasticity".

## Paramters & Command Line Arguments

- num_episodes: 
  - Number of episodes per run.
  - Default value: 1000

- discount_factor:
    - Factor that controls how much we care about future actions.
    - Default value: 1.0

- alpha:
    - Update step size
    - Default: 0.5

- epsilon:
    - Probability of picking random action.
    - Default value: 0.1

- epochs:
    - Number of runs we want to average the plots over.
    - Default value: 10

- average_over_n:
    - Smooth the graph by averaging over every n episodes.
    - Default value: 50

- extra_actions:
    - To set extra_actions to True and thus incorperate 8 actions: --extra_actions
    - To set extra_actions to False and thus incorperate 4 actions: --no_extra_actions
    - Default value: False

- add_stochasticity:
    - To set add_stochasticity to True and thus incorperate stochasticity: --add_stochasticity
    - To set add_stochasticity to False and thus incorperate stochasticity: --no_add_stochasticity
    - Default value: False
