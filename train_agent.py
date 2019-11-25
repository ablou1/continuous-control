from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from agent import ActorCriticAgent
import numpy as np
from train import ddpg

# Load the environment
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

# Create the agent to train with the parameters to use
agent = ActorCriticAgent(state_size=state_size, action_size=action_size, seed=0)

# Run the training
scores_mean_agent, score_mean_last100 = ddpg(env, agent, num_agents, brain_name, n_episodes=200, save_checkpoint=True, simu_name='single_train')

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_mean_agent)), scores_mean_agent)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# plot the scores mean over the last 100 iterations
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(score_mean_last100)), score_mean_last100)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


env.close()
