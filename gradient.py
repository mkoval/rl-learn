#!/usr/bin/env python

from copy import deepcopy
from math import exp
from random import gauss

class Environment:
	def __init__(self, s0, smin, smax):
		self.s = s0
		self.smin = smin
		self.smax = smax

	def GetDims(self):
		return len(self.w)

	def GetState(self):
		return deepcopy(self.s)

	def DoAction(self, a):
		snew   = self.s + a
		snew   = min(max(snew, self.smin), self.smax)
		a      = snew - self.s
		self.s = snew
		r      = -pow(snew, 2) - pow(a, 2)
		return (a, r, snew)
