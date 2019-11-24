from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import ActorCriticAgent
import torch
import random

# Load the Banana environment
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)


action_size = brain.vector_action_space_size    # size of each action
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
#print('The state for the first agent looks like:', states[0])

# Create the agent to train with the parameters to use
agent = ActorCriticAgent(state_size=state_size, action_size=action_size, seed=0)


def ddpg(agent, n_episodes=2000, max_t=1000, save_checkpoint=False):
    """DDPG.

    Params
    ======
        agent (Agent): agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    best_score = 30.0                  # Only save the agent if he gets a result better than 30.0
    # Number of episodes needed to solve the environment (mean score of 13 on the 100 last episodes)
    episode_solved = n_episodes
    scores_mean_agent = []             # list containing scores from each episode
    scores_mean_last100 = []           # List containing mean value of score_window
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        scores = np.zeros(num_agents)                      #initialize the score to 0            
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations              # get the current state
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        scores_window.append(scores.mean())                # save most recent score
        scores_mean_agent.append(scores.mean())            # save most recent score
        scores_mean_last100.append(np.mean(scores_window))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores.mean()), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= best_score:
            episode_solved = min(episode_solved, i_episode-100)
            best_score = np.mean(scores_window)
            if save_checkpoint:
                checkpoint = {'state_size': agent.state_size,
                              'action_size': agent.action_size,
                              'hidden_layer_size': agent.hidden_layer_size,
                              'state_dict': agent.qnetwork_local.state_dict()
                              }
                torch.save(checkpoint, f'{agent.name}_checkpoint.pth')
    if episode_solved < n_episodes:
        print(f'\n{agent.name} - best average score : {best_score} - Environment solved after {episode_solved} episodes')
    return scores_mean_agent, scores_mean_last100


scores, _ = ddpg(agent, n_episodes=1000, save_checkpoint=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


env.close()
