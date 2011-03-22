
class Agent:
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
