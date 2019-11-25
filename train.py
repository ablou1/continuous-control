# File dedicated to train a DDPG agent
from collections import deque
import numpy as np
import torch

def ddpg(env, agent, num_agents, brain_name, n_episodes=200, save_checkpoint=False, simu_name='test_DDPG'):
    """Method dedicated to train a DDPG agent.

    Params
    ======
        env (UnityEnvironment): The environment
        agent (ActorCriticAgent): agent to train
        num_agents (int): number of agnet trained at the same time
        brain_name (env.brain_names): brain link to the environment
        n_episodes (int): maximum number of training episodes
        save_checkpoint: indicate if the networks weights should be saved or not
    """
    best_score = 30.0                  # Only save the agent if he gets a result better than 30.0
    # Number of episodes needed to solve the environment (mean score of 30 on the 100 last episodes)
    episode_solved = n_episodes
    scores_mean_agent = []             # list containing mean scores (over the 20 agents) from each episode
    scores_mean_last100 = []           # List containing mean value (over the 20 agents) of score_window
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        scores = np.zeros(num_agents)                      # initialize the score to 0            
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
        
        agent.end()   # update the agent at the end of the episode 

        scores_window.append(scores.mean())                # save most recent mean score
        scores_mean_agent.append(scores.mean())            # save most recent mean score
        scores_mean_last100.append(np.mean(scores_window))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores.mean()), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # check if the environment is solved
        if (i_episode > 100) and (np.mean(scores_window) >= best_score):
            episode_solved = min(episode_solved, i_episode-100)
            best_score = np.mean(scores_window)

            # save the networks weights
            if save_checkpoint:
                fc1_units, fc2_units = agent.critic_local.get_hiddenlayer_size()
                checkpoint_critic = {'state_size': agent.state_size,
                                     'action_size': agent.action_size,
                                     'fc1_units': fc1_units,
                                     'fc2_units': fc2_units,
                                     'critic_state_dict': agent.critic_local.state_dict()
                                    }

                fc1_units, fc2_units = agent.actor_local.get_hiddenlayer_size()
                checkpoint_actor = {'state_size': agent.state_size,
                                    'action_size': agent.action_size,
                                    'fc1_units': fc1_units,
                                    'fc2_units': fc2_units,
                                    'actor_state_dict': agent.actor_local.state_dict()
                                   }
                torch.save(checkpoint_critic, f'{simu_name}_critic_checkpoint.pth')
                torch.save(checkpoint_actor, f'{simu_name}_actor_checkpoint.pth')

                if episode_solved < n_episodes:
                    print(f'\n{simu_name} - best average score : {best_score} - Environment solved after {episode_solved} episodes')

                return scores_mean_agent, scores_mean_last100

    if episode_solved < n_episodes:
        print(f'\n{simu_name} - best average score : {best_score} - Environment solved after {episode_solved} episodes')
    return scores_mean_agent, scores_mean_last100
