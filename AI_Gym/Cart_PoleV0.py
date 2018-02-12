"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from Brain_DQN import DQN_linear

env = gym.make('CartPole-v0')
env = env.unwrapped
max_episodes=1000
max_steps=500

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DQN_linear(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, exploration=0.1,reward_decay=0.9,
                  replace_target_iterations=100, memory_size=2000,batch_size=32,
                  exploration_decrement=0.99,output_graph=True)

total_steps = 0

for i_episode in range(max_episodes):

    observation = env.reset()
    ep_r = 0
    step=1
    while True:
        # if i_episode%20==0:
        env.render()
        action = RL.choose_action(observation,env)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, done, observation_)

        ep_r += reward
        if total_steps > 1000:
          RL.learn()

        if done or step>=max_steps:
            # print('episode: ', i_episode,
            #       'ep_r: ', round(ep_r, 2),
            #       ' epsilon: ', round(RL.epsilon, 2),end='')
            print ("\rEpisode {} - ep_r {} - step {} -epsilon {}".format(i_episode, round(ep_r,2),step, round(RL.epsilon,2)), end="")
            step=0
            break

        observation = observation_
        step+=1
        total_steps += 1