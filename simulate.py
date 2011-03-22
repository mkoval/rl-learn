#!/usr/bin/env python

from __future__ import print_function
import csv
import copy
import optparse
import numpy
import random
import sys

import agent_qlearning
import world_grid

QLearningAgent = agent_qlearning.Agent
State = world_grid.State
World = world_grid.World

def Simulate(world, agent, episodes):
	rewards = numpy.zeros(episodes)

	for episode in range(0, episodes):
		state = copy.deepcopy(world)

		while not state.IsTerminal():
			action = agent.Deliberate(state)
			future, reward = state.Act(action)
			agent.Learn(state, action, future, reward)

			state             = future
			rewards[episode] += reward
	
	return rewards

def main(argv):
	if len(argv) <= 4:
		print('err: incorrect number of arguments', file=sys.stderr)
		print('usage: ./simulate alpha gamma n it', file=sys.stderr)
		return 1

	# Parse the command-line parameters of alpha, beta, and n.
	try:
		alpha = float(argv[1])
		gamma = float(argv[2])
		if not 0 <= alpha <= 1.0 or not 0 <= gamma <= 1.0:
			raise Exception()
	except:
		print('err: alpha and gamma must be between zero and one', file=sys.stderr)
		return 1
	
	try:
		n  = int(argv[3])
		it = int(argv[4])
		if n < 0 or it < 0:
			raise Exception()
	except:
		print('err: number of iterations must be a positive integer', file=sys.stderr)
		return 1

	# Grid world depicted on p.646 of Russel and Norvig (3rd Ed.).
	world = World(4, 3)
	world.AddObstacle(1, 1)

	# Receive a reward of -0.04 each move with a probablity of 0.20 of moving
	# perpendicular to the intended direction.
	state = State(world, 0, 0, 0.20, -0.04)
	state.AddReward(3, 2, +1)
	state.AddReward(3, 1, -1)
	state.AddTerminal(3, 2)
	state.AddTerminal(3, 1)

	# Learn the optimal policy using Q-Learning with alpha = gamma = 0.20.
	time        = numpy.arange(0, n)
	mean_reward = numpy.zeros(n)

	for i in range(0, it):
		agent = QLearningAgent(alpha, gamma, 0.0)
		mean_reward += Simulate(state, agent, n)

	mean_reward /= it

	# Save the output as a CSV file for analysis and/or plotting.
	writer  = csv.writer(sys.stdout, delimiter='\t')
	writer.writerows(zip(time, mean_reward))
	return 0

if __name__ == '__main__':
	try:
		sys.exit(main(sys.argv))
	except KeyboardInterrupt:
		pass

