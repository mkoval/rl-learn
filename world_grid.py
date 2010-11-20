from __future__ import print_function
import copy
import numpy
import random

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

