import numpy as np

class Dropout(object):
	def __init__(self, p=0.5):
		# Dropout probability
		self.p = p
		self.mask = np.zeros(None)

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x, train = True):
		# 1) Get and apply a mask generated from np.random.binomial
		# 2) Scale your output accordingly
		# 3) During test time, you should not apply any mask or scaling.
		if train:
			self.mask = np.random.binomial(1,self.p,x.shape)
			#Invert the mask
			return np.multiply(x,self.mask)/self.p                                                                           
		return x


	def backward(self, delta):
		print(delta)
		return np.multiply(delta,self.mask)
