#!/usr/bin/env python
# coding: utf-8

# # Planning
# 
# Planning is a method of simulating a sequence of actions in an environment model before actually taking an action in the real environment.
# 
# Concepts covered:
# 1. Cross entropy method
# 2. Monte Carlo tree search
# 3. Probabilistic ensembles

# ## Cross Entropy Method
# 
# The Cross Entroy Method (CEM) is a gradient-free optimization method commonly used for planning in model-based reinforcement learning.
# 
# CEM Algorithm
# 1. Create a Gaussian distribution $N(\mu,\sigma)$ that describes the weights $\theta$ of the neural network.
# 2. Sample $N$ batch samples of $\theta$ from the Gaussian.
# 3. Evaluate all $N$ samples of $\theta$ using the value function, e.g. running trials.
# 4. Select the top % of the samples of $\theta$ and compute the new $\mu$ and $\sigma$ to parameterise the new Gaussian distribution.
# 5. Repeat steps 1-4 until convergence.

# In[1]:


import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import gym
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# RL Gym
env = gym.make('CartPole-v1')

# Initialisation
n = 10  # number of candidate policies
top_k = 0.40  # top % selected for next iteration
mean = np.zeros((5,2))  # shape = (n_parameters, n_actions)
stddev = np.ones((5,2))  # shape = (n_parameters, n_actions)


# In[3]:


def get_batch_weights(mean, stddev, n):
    mvn = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_diag=stddev)
    return mvn.sample(n).numpy()

def policy(obs, weights):
    return np.argmax(obs @ weights[:4,:] + weights[4])

def run_trial(weights, render=False):
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        a = policy(obs, weights)
        obs, r, done, _ = env.step(a)
        reward += r
        if render:
            env.render()
    return reward

def get_new_mean_stddev(rewards, batch_weights):
    idx = np.argsort(rewards)[::-1][:int(n*top_k)]
    mean = np.mean(batch_weights[idx], axis=0)
    stddev = np.sqrt(np.var(batch_weights[idx], axis=0))
    return mean, stddev


# In[4]:


for i in range(20):
    batch_weights = get_batch_weights(mean, stddev, n)
    rewards = [run_trial(weights) for weights in batch_weights]
    mean, stddev = get_new_mean_stddev(rewards, batch_weights)
    print(rewards)


# In[5]:


mean, stddev


# In[6]:


best_weights = get_batch_weights(mean, stddev, 1)[0]


# In[7]:


run_trial(best_weights, render=False)


# ## Monte Carlo Tree Search
# 
# Upcoming

# ## Probabilistic Ensembles
# 
# Upcoming
