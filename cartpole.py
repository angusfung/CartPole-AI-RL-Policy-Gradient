#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import sys



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()



env = gym.make('CartPole-v0') #Change 1

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

hidden_size = 2 #Change 2
alpha = 0.0001  #Change 3
TINY = 1e-8
gamma = 0.99 #Change 4

weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

if args.load_model:
    model = np.load(args.load_model)
    hw_init = tf.constant_initializer(model['hidden/weights'])
    hb_init = tf.constant_initializer(model['hidden/biases'])
    mw_init = tf.constant_initializer(model['mus/weights'])
    mb_init = tf.constant_initializer(model['mus/biases'])
    sw_init = tf.constant_initializer(model['sigmas/weights'])
    sb_init = tf.constant_initializer(model['sigmas/biases'])
else:
    hw_init = weights_init
    hb_init = relu_init
    mw_init = weights_init
    mb_init = relu_init
    sw_init = weights_init
    sb_init = relu_init

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4 
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.int32, shape=(None, output_units), name='y') #Change 5



hidden = fully_connected( #Change 6
    inputs=x,
    num_outputs=hidden_size,
    activation_fn=tf.nn.softmax,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='hidden')

all_vars = tf.global_variables()

pi = tf.contrib.distributions.Bernoulli(p=hidden, name='pi') #Change 7
pi_sample = pi.sample()
log_pi = pi.log_prob(y, name='log_pi')

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

steps = []

track_returns = []
for ep in range(5000):

    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        #env.render()
        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0]) #Change 8
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    if not args.load_model:
        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
        
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.array(ep_actions),
                                Returns:returns })
    steps.append(t)
    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))


    # with tf.variable_scope("mus", reuse=True):
    #     print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:])


sess.close()
tot_epis = 5000
x_axis = linspace(0, tot_epis, len(steps))
plt_steps = plt.plot(x_axis, steps)
plt.xlabel('Episode Number')
plt.ylabel('Number of Steps')
plt.title('Performance')
plt.show()
plt.savefig("cartpole.png")
