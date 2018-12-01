'''equation generator '''

import numpy as np 
import random


max_num = 99
decimal = True
dec_limit = 3

sym_order = ['?','+','-','*','/','=','(',')','.','0','1','2','3','4','5','6','7','8','9']
sym_num = len(sym_order)


def number_generator(max_num=max_num,dec_limit= dec_limit,decimal = decimal):
	#function returns a random number tuple, (string,float) can be a decimal number if keyword true
	num = np.random.choice(list(range(max_num+1)))
	num_str = str(num)
	if decimal:
		dec_len = np.random.choice(list(range(1,dec_limit)))
		dec_str = ''
		for i in range(dec_len):
			place = np.random.choice(list(range(10)))
			dec_str+= str(place)
		dec = int(dec_str)
		num_str = num_str+'.'+dec_str
		num = float(num_str)
	return num_str,num

def expression_generator(max_len = 6,max_num = max_num,dec_limit=dec_limit,decimal=decimal, algebraic = False, max_paren = 3):
	operators = ['+','-','*','/']
	exp_len = np.random.choice(list(range(max_len+1)))
	nums = []
	ops = []
	exp_ls = []
	for i in range(exp_len):
		nums.append(number_generator(max_num=max_num,dec_limit= dec_limit,decimal = decimal))
	for i in range(exp_len-1):
		ops.append(np.random.choice(operators))
		exp_ls.append(nums[i])
		exp_ls.append(ops[i])
	exp_ls.append(nums[-1])
	num_paren = np.random.choice(list(range(max_paren+1)))
	parens = {}
	for i in range(num_paren):
		start_time = time.time()
		okay = False
		while not okay:
			time = time.time() - start_time
			if time > .1:
				break
			left = np.random.choice(list(range(len(nums)-1)))
			right = np.random.choice(list(range(left,len(nums)+1)))
			for pair in parens:
				if left == pair[0] and right == pair[1]:
					okay == False
					break
				if left >= pair[0] and right > pair[1]:
					okay == False
					break
				if right <= pair[1] and left < pair[0]:
					okay == False
					break
			okay == True
		if okay = True:
			parens[i] = [left,right]
	for pair in parens:
		for p in range(2):
			position = -1
			for i in exp_ls:
				if exp_ls[i] in nums:
					position += 1
					if pair[p] == position:
						if exp_ls[i-1] == '(' and p == 1:
							exp_ls.insert(i-1,')')
						elif exp_ls[i+1] == ')' and p == 0:
							exp_ls.insert(i+1,'(')
				






def two_num_operate(num1,num2,operator):
	stringed = '%s%s%s'%(str(num1),str(operator),str(num2))
	return stringed,exec(stringed)





def algebraic_equation():
	return True

def well_formed_checker(input_string):
	return True

def string_generator(type = 'none',symbols = sym_order):
	return '1+1=2'


