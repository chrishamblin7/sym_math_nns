#! /usr/bin/env python
from pygame.locals import *
import os
import random
import pygame
import math
import sys
import numpy as np
from math import pi, cos, sin
from copy import deepcopy
from subprocess import call
import time
import scipy.misc

human = False

symbols = ['!','.','0','1','2','3','4','5','6','7','8','9','?','+','-','*','/','(',')','a','b','c','d','x','y','z','<','>','=']

max_symbols = 10
max_size = 60
min_size = 15
max_dim = 256
border_buffer = 50
pic_num = 20000
fontfile = open('font_list.txt','r')
fullfontlist = [x.strip() for x in fontfile.readlines()]


if not human:
	outputfile = open('symbol_images/image_whiteonblack_info.csv','w+')
	outputfile.write('pic_num,sym_num,symbol,size,position,color,font\n')


os.environ["SDL_VIDEO_CENTERED"] = "1"

pygame.init()

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE = (  0,   0, 255)


screen = pygame.display.set_mode((max_dim,max_dim))

pygame.display.set_caption("Line Up Shapes")

pygame.font.init() # you have to call this at the start, 


clock = pygame.time.Clock()

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


def update_parameters(pic_num = 0,max_symbols = max_symbols,max_size = max_size, min_size = min_size,symbols = symbols,human = human,fullfontlist = fullfontlist, max_dim = max_dim,screen=screen):
	start = time.time()
	num_symbols = int(np.random.choice(list(range(1,max_symbols+1))))
	#num_symbols = 1
	#background_color = gen_color(avoid_color = 'none')
	background_color = BLACK
	screen.fill(background_color)
	positions = []
	if human:
		toprint = 'symbol    size          position           color                font\n'
	else:
		toreturn = ''
	for s in range(num_symbols):
		symbol = np.random.choice(symbols)
		symbol_size = int(np.random.choice(list(range(min_size,max_size))))
		symbol_font_name = np.random.choice(fullfontlist)
		#symbol_font_name = fullfontlist[font_num]
		#font_num += 1
		#color = gen_color(avoid_color = background_color)
		color = WHITE
		font = pygame.font.SysFont(symbol_font_name, symbol_size)
		fontsurface = font.render(symbol, False, color)
		surface_size = fontsurface.get_width(), fontsurface.get_height()
		# positioning
		spaced = False
		while not spaced:
			spaced = True
			x = int(np.random.choice(list(range(border_buffer,max_dim-border_buffer))))
			y = int(np.random.choice(list(range(border_buffer,max_dim-border_buffer))))
			for p in positions:
				if (abs(x - p[0][0]) < surface_size[0]/2 + p[1][0]/2) or (abs(y - p[0][1]) < surface_size[1]/2 + p[1][1]/2):
					spaced = False
					break
			finish = time.time()
			if finish - start > .5:
				return False, toreturn

		positions.append([(x,y),surface_size])		
		screen.blit(fontsurface,positions[s][0])
		if human:
			toprint += '%s         %s       %s       %s          %s\n'%(symbol,surface_size,positions[s][0],color,symbol_font_name)
		else:
			toreturn += ','.join((str(pic_num),str(s+1),str(symbol),str(surface_size),str(positions[s][0]),str(color),str(symbol_font_name)+'\n'))
	if human:		
		print(toprint)
		return True	
	else:
		return True, toreturn

if human:
	running = True 
	while running:
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()

			key = pygame.key.get_pressed()


			#utility keys
			#update
			if key[pygame.K_u]:
				success = False
				while not success:
					success = update_parameters()

			pygame.display.update()


		clock.tick(40)


else:
	#for pic in range(pic_num):
	for pic in range(120000,120000+pic_num):
		success = False
		while not success:
			success, new_rows = update_parameters(pic_num=pic)
		outputfile.write(new_rows)
		outputfile.flush()
		pygame.image.save(screen, "symbol_images/%s_symbols.png"%pic)
