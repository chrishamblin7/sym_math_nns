''' Data manipulation functions '''
import numpy as np
import torch
from torch import from_numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def num_2_onehot(num, start, stop):
    onehot = []
    for i in range(start, stop+1):
        if i == num:
            onehot.append(1.0)
        else:
            onehot.append(0.0)
    arr = np.array(onehot)
    return arr


def onehot_2_gaussian(x, std = .4, samples = 200):
	'''takes a one-hot encoded pytorch tensor or numpy array and adds noise around the encoded number'''
	count = {}
	for i in range(len(x)):
		count[i] = 0

	mean = int(np.where(x==1)[0])
	for n in range(samples):
		out_of_bounds = True
		while out_of_bounds:
			i = int(np.round(np.random.normal(mean,std,1)))
			if -1 < i < len(x):
				out_of_bounds = False
		count[i] += 1
		#print(count)
	ls = []
	for i in range(len(x)):
		ls.append(float(count[i])/float(samples))
	#print(ls)
	return np.array(ls)

def torch_vec_2_classlabel(tens):
	_, labels = tens.max(dim=1)
	return labels