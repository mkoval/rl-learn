import sys

n = 10

class GridState:
	def __init__(self, size, pos):
		self.rows = size[0]
		self.cols = size[1]
		self.x    = pos[0]
		self.y    = pos[1]

class GridAgent:
	def __init__(self, alpha, gamma):
		self.alpha = alpha
		self.gamma = gamma

	def Perceive(self, state):
		return state
	
	def Deliberate(self, percept):
		return 0
	
	def Act(self, state, action):
		return (state, 1)
	
	def Learn(self, state, action, future, reward):
		pass

def Simulate(state, agent):
	# Decide on an action to perform on the world.
	percept = agent.Perceive(state)
	action  = agent.Deliberate(percept)

	# Act on the environment and learn from the reward.
	future, reward = agent.Act(state, action)
	agent.Learn(state, action, future, reward)

	return (action, future, reward)

def main(argv):
	agent = GridAgent(0.2, 0.2)
	world = 0

	states    = [ None ] * (n + 1)
	actions   = [ None ] * n
	rewards   = [ None ] * n
	states[0] = world

	for t in range(0, n):
		actions[t], states[t + 1], rewards[t] = Simulate(states[t], agent)

	print('Total Reward   = {0}'.format(sum(rewards)))
	print('Average Reward = {0}'.format(sum(rewards) / n))
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
