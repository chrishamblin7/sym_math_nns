''' symbolic data generators for arithmetic '''
import numpy as np
import pickle
import sys
sys.path.append('/hd1/scsnl/neural_networks/modular_math_cog/utility')
import data_modifiers
from PIL import Image
import os
from os.path import join as pjoin
from torchvision import datasets, transforms, utils
np.set_printoptions(threshold=np.inf)
import torch
from torch import from_numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pygame.locals import *
import random
import pygame
from math import pi, cos, sin
from copy import deepcopy
from subprocess import call
import time
import scipy.misc

def dataset_dict_generator(samples = 2000, nums = [0,1,2,3,4,5,6,7,8,9],ops = ['+','-'], min_ans = 0, max_ans = 18, noise = .5):
	output_dict = {}
	for p in range(samples):
		output_dict[p] = {}
		in_range = False
		while not in_range:
			num1 = np.random.choice(nums)
			num2 = np.random.choice(nums)
			op = np.random.choice(ops)
			string = str(num1)+op+str(num2)
			ans = eval(str(num1)+op+str(num2))
			if min_ans <= ans <= max_ans:
				in_range = True
				output_dict[p]['num1'] = num1
		output_dict[p]['num2'] = num2
		output_dict[p]['op'] = op
		output_dict[p]['op_onehot'] = data_modifiers.num_2_onehot(ops.index(op),0,len(ops)-1)
		output_dict[p]['string'] = string
		output_dict[p]['ans'] = ans
		output_dict[p]['num1_onehot'] = data_modifiers.num_2_onehot(num1,min(nums),max(nums))
		output_dict[p]['num2_onehot'] = data_modifiers.num_2_onehot(num2,min(nums),max(nums))
		output_dict[p]['ans_onehot'] = data_modifiers.num_2_onehot(num2,min_ans,max_ans)
		output_dict[p]['num1_gauss'] = data_modifiers.onehot_2_gaussian(output_dict[p]['num1_onehot'],std = noise)
		output_dict[p]['num2_gauss'] = data_modifiers.onehot_2_gaussian(output_dict[p]['num2_onehot'],std = noise)
		output_dict[p]['ans_gauss'] = data_modifiers.onehot_2_gaussian(output_dict[p]['ans_onehot'],std = noise)

	file_name = str(samples)+'_'
	for o in ops:
		file_name += o
	file_name += '_'+str(noise)
	pickle.dump(output_dict,open('../data/datasets/%s.pkl'%file_name,'wb'))



def gen_color(avoid_color = (0,0,0)):
	redo = True
	while redo:
		redo = False
		r = int(np.random.choice(list(range(255))))
		g = int(np.random.choice(list(range(255))))
		b = int(np.random.choice(list(range(255))))
		#print('%s %s %s'%(r,g,b))
		if avoid_color != 'none':
			if abs(r - avoid_color[0]) < 40 and abs(g - avoid_color[1]) < 40 and abs(b - avoid_color[2]) < 40: 
				redo = True
	return (r,g,b)



def image_generator(samples_per = 100, nums = [0,1,2,3,4,5,6,7,8,9],ops = ['+','-'],min_ans = 0,max_ans = 18,
max_size = 40,min_size = 15,max_dim = 256,border_buffer = 30, versions = ['color','bow','wob'],
fontpath = '/hd1/scsnl/neural_networks/modular_math_cog/utility/font_list.txt',outputdir = '../data/images'):
	fontfile = open(fontpath,'r')
	fullfontlist = [x.strip() for x in fontfile.readlines()]
	BLACK = (  0,   0,   0)
	WHITE = (255, 255, 255)
	RED = (255,   0,   0)
	GREEN = (  0, 255,   0)
	BLUE = (  0,   0, 255)
	draw_space = int(np.floor(float(max_dim - 4*border_buffer)/3))

	os.environ["SDL_VIDEO_CENTERED"] = "1"
	pygame.init()
	screen = pygame.display.set_mode((max_dim,max_dim))
	pygame.display.set_caption("Simple Operation")
	pygame.font.init() 
	clock = pygame.time.Clock()

	output_dict = {}
	p = -1
	for num1 in nums:
		for num2 in nums:
			for op in ops:
				ans = eval(str(num1)+op+str(num2))
				if not 0 <= ans:
					continue
				for sample in range(samples_per):
					string = str(num1)+op+str(num2)
					symbols = {'num1':num1,'num2':num2,'op':op}

					for version in versions:
						colors = {}
						if version == 'color':
							background_color = gen_color(avoid_color = 'none')
							colors['num1'] = gen_color(avoid_color = background_color)
							colors['op'] = gen_color(avoid_color = background_color)
							colors['num2'] = gen_color(avoid_color = background_color)
						elif version == 'bow':
							background_color = WHITE
							colors['num1'] = BLACK
							colors['num2'] = BLACK
							colors['op'] = BLACK
						elif version == 'wob':
							background_color = BLACK
							colors['num1'] = WHITE
							colors['num2'] = WHITE
							colors['op'] = WHITE
						screen.fill(background_color)

						positions = {}
						positions['num1'] = (int(np.random.choice(list(range(border_buffer,border_buffer+draw_space)))),int(np.random.choice(list(range(border_buffer,max_dim-border_buffer)))))
						positions['op'] = (int(np.random.choice(list(range(2*border_buffer+draw_space,2*border_buffer+2*draw_space)))),int(np.random.choice(list(range(border_buffer,max_dim-border_buffer)))))
						positions['num2'] = (int(np.random.choice(list(range(3*border_buffer+2*draw_space,3*border_buffer+3*draw_space)))),int(np.random.choice(list(range(border_buffer,max_dim-border_buffer)))))

						sizes = {}
						fonts  = {}
						for sym in ['num1', 'op', 'num2']: 
							sizes[sym] = int(np.random.choice(list(range(min_size,max_size))))
							fonts[sym] = np.random.choice(fullfontlist)
						#drawing
						for sym in ['num1', 'op', 'num2']: 
							font = pygame.font.SysFont(fonts[sym], sizes[sym])
							fontsurface = font.render(str(symbols[sym]), False, colors[sym])
							surface_size = fontsurface.get_width(), fontsurface.get_height()		
							screen.blit(fontsurface,positions[sym])

						output_file_name = str(num1)+op+str(num2)+'_'+version+'_'+str(sample)+'.png'
						pygame.image.save(screen, pjoin(outputdir,output_file_name))

						p += 1
						output_dict[p] = {}
						output_dict[p]['num2'] = num2
						output_dict[p]['op'] = op
						output_dict[p]['op_onehot'] = data_modifiers.num_2_onehot(ops.index(op),0,len(ops)-1)
						output_dict[p]['string'] = string
						output_dict[p]['ans'] = ans
						output_dict[p]['num1_onehot'] = data_modifiers.num_2_onehot(num1,min(nums),max(nums))
						output_dict[p]['num2_onehot'] = data_modifiers.num_2_onehot(num2,min(nums),max(nums))
						output_dict[p]['ans_onehot'] = data_modifiers.num_2_onehot(num2,min_ans,max_ans)
						output_dict[p]['symbols'] = symbols
						output_dict[p]['colors'] = colors
						output_dict[p]['sizes'] = sizes
						output_dict[p]['positions'] = positions
						output_dict[p]['ans'] = ans
						output_dict[p]['file_name'] = output_file_name
	pickle.dump(output_dict,open(pjoin(outputdir,'image_dict.pkl'),'wb'))



class CountingDotsDataSet(Dataset):
	"""How many dots are in this pic dataset."""

	def __init__(self, root_dir, train=True, blackandwhite=False, cutoff=10, resize = False, size=30, include_negatives = True, onehot=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])):
		"""
		"""
		
		self.root_dir = root_dir
		if train:
			self.pic_dir = os.path.join(root_dir,'train')
		else:
			self.pic_dir = os.path.join(root_dir,'test')
		self.pic_names = sorted(os.listdir(self.pic_dir))
		if not include_negatives:
			self.pic_names = [i for i in self.pic_names if not ('neg' in i)]
		self.transform = transform
		self.onehot = onehot
		self.size = size
		self.resize = resize
		self.cutoff = cutoff
	def __len__(self):
		return len(self.pic_names)

	def get_label_from_name(self,img_name,cutoff):
		split_name = img_name.split('_')
		num = int(split_name[0])-1
		if num > int(cutoff-1):
			num = int(cutoff-1)
		if self.onehot:
			return numtoarr(num,1,cutoff)
		else:
			return np.array(num)       

	def __getitem__(self, idx):
		img_name = os.path.join(self.pic_dir,self.pic_names[idx])
		#image = image.imread(img_name,as_gray=True)
		#image = rgb2grey(image)
		image = Image.open(img_name)
		if self.resize:
			image = image.resize((self.size,self.size))	
		#image = np.array([image])
		#image = image.astype(float)
		label = self.get_label_from_name(self.pic_names[idx],self.cutoff)
		#sample = {'image': image, 'label': label}
		if self.transform:
			image = self.transform(image)
		label = from_numpy(label)
		sample = (image,label)
		return sample


class three_num_abstract(Dataset):
	def __init__(self, dict_path, train=True, train_test_split = .8, input_noise = True, ans_noise = False):
			self.dict_path = dict_path
			self.dict = pickle.load(open(dict_path,'rb'))
			self.train = train
			self.train_test_split = train_test_split
			self.input_noise = input_noise
			self.ans_noise = ans_noise
			self.file_name = dict_path.split('/')[-1]
			file_name_list = self.file_name.split('_')
			self.samples = int(file_name_list[0])
			self.noise_std = float(file_name_list[-1].replace('.pkl',''))
			self.training_samples = int(np.ceil(float(len(self.dict))*self.train_test_split))
			self.testing_samples = int(len(self.dict)-np.ceil(float(len(self.dict))*self.train_test_split))
	def __len__(self):
		if self.train:
			return self.training_samples
		else:
			return self.testing_samples

	def __getitem__(self, idx):
		if self.train:
			p = idx
		else:
			p = int(np.ceil(float(len(self.dict))*self.train_test_split))+idx
		if self.input_noise:
			np_input = np.concatenate((self.dict[p]['num1_gauss'],self.dict[p]['num2_gauss'],self.dict[p]['op_onehot']))
		else:
			np_input = np.concatenate((self.dict[p]['num1_onehot'],self.dict[p]['num2_onehot'],self.dict[p]['op_onehot']))
		input = from_numpy(np_input).type(torch.FloatTensor)
		if self.ans_noise:
			label = from_numpy(self.dict[p]['ans_gauss']).type(torch.FloatTensor)
		else:
			label = from_numpy(self.dict[p]['ans_onehot']).type(torch.FloatTensor)
		return (input,label)

	def get_info(self):
		input_dim = len(np.concatenate((self.dict[0]['num1_gauss'],self.dict[0]['num2_gauss'],self.dict[0]['op_onehot'])))
		label_dim = len(self.dict[0]['ans_gauss'])
		samples = self.samples
		noise_std = self.noise_std
		return {'input_dim':input_dim,'label_dim':label_dim,'noise_std':noise_std,'samples':samples,'training_samples':self.training_samples,'testing_samples':self.testing_samples}


#class three_num_image(Dataset):