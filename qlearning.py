#!/usr/bin/env python

from __future__ import print_function
import copy
import random
import sys

class World:
	def __init__(self, width, height):
		self.width    = width
		self.height   = height
		self.passable = [ [  True  ] * width for y in range(0, height) ]

	def __key(self):
		# TODO: Include self.passable.
		return (self.width, self.height)
	
	def __eq__(self, state):
		return self.__key() == state.__key()

	def __hash__(self):
		return hash(self.__key())

	def AddObstacle(self, x, y):
		self.passable[y][x] = False

	def IsPassable(self, x, y):
		is_inbounds = 0 <= x < self.width and 0 <= y < self.height
		return is_inbounds and self.passable[y][x]
	
class State:
	def __init__(self, world, x, y, prob, reward):
		self.world    = world
		self.x        = x
		self.y        = y
		self.prob     = prob
		self.default  = reward
		self.rewards  = [ [ reward ] * world.width for y in range(0, world.height) ]
		self.terminal = [ [ False  ] * world.width for y in range(0, world.height) ]

	def __key(self):
		# TODO: Include self.rewards.
		return (self.world, self.x, self.y, self.prob)
	
	def __eq__(self, state):
		return self.__key() == state.__key()

	def __hash__(self):
		return hash(self.__key())

	def AddReward(self, x, y, value):
		self.rewards[y][x] = value
	
	def AddTerminal(self, x, y):
		self.terminal[y][x] = True

	def GetReward(self):
		return self.rewards[self.y][self.x]

	def GetActions(self):
		return [ 'L', 'R', 'U', 'D' ]
	
	def IsTerminal(self):
		return self.terminal[self.y][self.x]

	def Act(self, action):
		# Express movement as the complex number x + y*i with a probability of
		# p to move orthogonal to the desired direction of movement.
		delta = {
			'L' : -1,
			'R' :  1,
			'U' :  1j,
			'D' : -1j
		}[action]

		seed = random.uniform(0, 1)
		if 0.0 <= seed < self.prob / 2.0:
			delta *=  1j
		elif self.prob / 2.0 <= seed < self.prob:
			delta *= -1j

		# Restrict movement to open squares. Hitting a wall incurs the same
		# reward as landing on the square normally.
		state_next = copy.deepcopy(self)

		if self.world.IsPassable(self.x + int(delta.real), self.y + int(delta.imag)):
			state_next.x += int(delta.real)
			state_next.y += int(delta.imag)
			reward        = self.rewards[state_next.y][state_next.x]
		else:
			reward = self.default

		return (state_next, reward)

class QLearningAgent:
	def __init__(self, alpha, gamma, start):
		self.q       = dict()
		self.alpha   = alpha
		self.gamma   = gamma
		self.start   = start

	def SetValue(self, state, action, value):
		self.q[(state, action)] = value

	def GetValue(self, state, action):
		if (state, action) in self.q:
			return self.q[(state, action)]
		else:
			return self.start

	def Deliberate(self, state):
		return max(state.GetActions(), key=lambda a: self.GetValue(state, a))

	def Learn(self, state, action, future, reward):
		q_old = self.GetValue(state, action)
		q_max = max([ self.GetValue(future, a) for a in state.GetActions() ])

		q_new = q_old + self.alpha * (reward + self.gamma * q_max - q_old)

		self.SetValue(state, action, q_new)

def Simulate(state, agent):
	action = agent.Deliberate(state)
	future, reward = state.Act(action)
	agent.Learn(state, action, future, reward)
	return (future, action, reward)

def main(argv):
	if len(argv) <= 1:
		print('err: incorrect number of arguments', file=sys.stderr)
		print('usage: ./rl ticks', file=sys.stderr)
		return 1

	ticks = int(argv[1])

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
	agent = QLearningAgent(0.2, 0.2, 1.0)

	states    = [ state ]
	actions   = [ ]
	rewards   = [ ]

	while not state.IsTerminal():
		state, action, reward = Simulate(state, agent)
		states.append(state)
		actions.append(action)
		rewards.append(reward)

	print('Total Reward   = {0}'.format(sum(rewards)))
	print('Average Reward = {0}'.format(sum(rewards) / len(rewards)))
	return 0

if __name__ == '__main__':
	try:
		sys.exit(main(sys.argv))
	except KeyboardInterrupt:
		pass

