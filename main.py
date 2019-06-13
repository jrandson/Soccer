from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt
import torch

from agent import Agent, Replay


def run_ddpg(env, goalier_team, striker_team, max_episodes=10000, max_steps=300):
    save_every = 10
    scores = []
    goalier_brain_name, striker_brain_name = env.brain_names

    for episode in range(1, max_episodes + 1):

        episode_score = 0

        env_info = env.reset(train_mode=True)

        goalier_state = env_info[goalier_brain_name].vector_observations
        goalier_state = np.reshape(goalier_state, (1, -1))

        striker_state = env_info[striker_brain_name].vector_observations
        striker_state = np.reshape(striker_state, (1, -1))

        for step_i in range(max_steps):
            print("\t\r Step {} from {}\t ".format(step_i, max_steps), end="", flush=True)

            goalier_actions = np.asarray([agent.act(goalier_state, add_noise=True) for agent in goalier_team])
            # print("goalier_action: ", goalier_actions)
            striker_actions = np.asarray([agent.act(striker_state, add_noise=True) for agent in striker_team])
            # print("striker_actions", striker_actions)

            discrete_goalier_actions = np.asarray([np.argmax(_, axis=1)[0] for _ in goalier_actions])
            discrete_striker_actions = np.asarray([np.argmax(_, axis=1)[0] for _ in striker_actions])

            # if np.random.rand() < 0.05:
            #     discrete_goalier_actions = np.random.randint(goalier_action_size, size=num_goalier_agents)
            #
            # if np.random.rand() < 0.05:
            #     discrete_striker_actions = np.random.randint(striker_action_size, size=num_striker_agents)

            # print("goalier:", discrete_goalier_actions)
            # print("striker:", discrete_striker_actions)

            actions = dict(zip([goalier_brain_name, striker_brain_name],
                               [discrete_goalier_actions, discrete_striker_actions]))

            env_info = env.step(actions)

            goalier_next_state = env_info[goalier_brain_name].vector_observations
            goalier_next_state = np.reshape(goalier_next_state, (1, -1))

            striker_next_state = env_info[striker_brain_name].vector_observations
            striker_next_state = np.reshape(striker_next_state, (1, -1))

            goalier_rewards = env_info[goalier_brain_name].rewards
            striker_rewards = env_info[striker_brain_name].rewards

            rewards = max(np.mean(goalier_rewards), np.mean(striker_rewards))

            goalier_dones = env_info[goalier_brain_name].local_done
            striker_dones = env_info[striker_brain_name].local_done

            episode_score += rewards

            # update each agent
            for i, goalier_agent in enumerate(goalier_team):
                goalier_agent.step(goalier_state, goalier_actions[i],
                                   goalier_rewards[i], goalier_next_state, goalier_dones[0])

            for i, striker_agent in enumerate(striker_team):
                striker_agent.step(striker_state, striker_actions[i], striker_rewards[i],
                                   striker_next_state, striker_dones[0])

            goalier_state = goalier_next_state
            striker_state = striker_next_state

            if np.any(goalier_dones):
                break

        scores.append(np.max(episode_score))
        mean_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)

        print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            episode, mean_score, episode_score))

        if mean_score >= 100:
            print("\t Model reached the score goal in {} episodes!".format(episode))
            break

        if episode % save_every == 0:
            # save goalier actor team
            for i, agent in enumerate(goalier_team):
                torch.save(agent.online_actor.state_dict(), "models/goalier_actor_%s.path" % i)
                torch.save(agent.online_critic.state_dict(), "models/goalier_critic_%s.path" % i)

            # save striker actor team
            for i, agent in enumerate(striker_team):
                torch.save(agent.online_actor.state_dict(), "models/striker_actor_%s.path" % i)
                torch.save(agent.online_critic.state_dict(), "models/striker_critic_%s.path" % i)

    return scores


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    env = UnityEnvironment(file_name="Soccer_Linux/Soccer.x86_64", no_graphics=False)
    # print(env)

    # set the goalie brain
    goalier_brain_name, striker_brain_name = env.brain_names

    goalier_brain = env.brains[goalier_brain_name]
    striker_brain = env.brains[striker_brain_name]

    env_info = env.reset(train_mode=True)  # reset the environment

    num_goalier_agents = len(env_info[goalier_brain_name].agents)
    num_striker_agents = len(env_info[striker_brain_name].agents)

    goalier_action_size = goalier_brain.vector_action_space_size
    striker_action_size = striker_brain.vector_action_space_size

    goalier_states = env_info[goalier_brain_name].vector_observations
    striker_states = env_info[striker_brain_name].vector_observations

    goalier_state_size = goalier_states.shape[1]
    striker_state_size = striker_states.shape[1]

    print("training the DDPG model")

    goalier_reply = Replay(goalier_action_size, buffer_size=int(1e6), batch_size=256)
    striker_reply = Replay(striker_action_size, buffer_size=int(1e6), batch_size=256)

    goalier_team = [Agent(state_size=goalier_state_size * 2, action_size=goalier_action_size, reply=goalier_reply)
                    for _ in range(num_goalier_agents)]
    striker_team = [Agent(state_size=striker_state_size * 2, action_size=striker_action_size, reply=striker_reply)
                    for _ in range(num_striker_agents)]

    print("Num goalier agent", len(goalier_team))
    print("Num striker agent", len(striker_team))

    scores = run_ddpg(env, goalier_team, striker_team)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_episodes.png")
    plt.show()

    w = 10
    mean_score = [np.mean(scores[i - w:i]) for i in range(w, len(scores))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(mean_score) + 1), mean_score)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_episodes_smorthed.png")
    plt.show()
