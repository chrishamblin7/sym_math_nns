##DataSet Generator for sequence to sequence mathematical symbol manipulators ###

import random
import numpy as np
import torch
import pickle
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

SOS_token = 0
EOS_token = 1

sym2index = {'0':2,'1':3,'2':4,'3':5,'4':6,'5':7,'6':8,'7':9,'8':10,'9':11,
                     '+':12,'-':13,'*':14,'.':15,'/':16,'(':17,')':18,'x':19,
                     'y':20,'z':21,'a':22,'b':23,'c':24,'=':25}

index2sym = dict((v,k) for k,v in sym2index.items())


def simple_num(dec = True, min_num = 0, max_num = 1000, max_float = 5):
    normal_num = str(np.random.randint(min_num,max_num))
    if dec:
        dec_length = np.random.randint(1,max_float)
        dec_num = ''
        for i in range(dec_length):
            dec_num = dec_num + str(np.random.randint(0,10))
        normal_num = normal_num + '.' + dec_num
    return normal_num

def simple_arith_pair(dec_pos = 'random'):
    num_operators = np.random.randint(1,5)
    ops = []
    dec = False
    if dec_pos == 'random':
        dec_pos = random.choice([True,False])
    str_valid = False
    while not str_valid:
        str_valid = True
        question_str = ''
        for i in range(num_operators):
            ops.append(random.choice(['+','-','*','/']))
        for i in range(len(ops)+1):
            if dec_pos:
                dec = random.choice([True,False])
            question_str = question_str + simple_num(dec = dec)
            if i != len(ops):
                question_str = question_str + ops[i]
        try:
            ans_str = str(('%f'%eval(question_str)).rstrip('0').rstrip('.'))
        except:
            str_valid = False

    return [question_str, ans_str]


def gen_arith_pairs(num = 10000,dec_pos = 'random'):
    pair_list = []
    char_list = []
    for i in range(num):
        new_pair = simple_arith_pair(dec_pos=dec_pos)
        for exp in new_pair:
            for char in exp:
                if char not in char_list:
                    char_list.append(char)

        pair_list.append(new_pair)
            
    return pair_list, len(char_list)

def gen_easy_train_test_split(num_range = range(101), ops = ['+','-','*'], never_seen = 10):
    num_list = list(num_range)
    #set aside some numbers that are only seen in the test set
    only_test_nums = []
    print(len(num_list))
    for i in range(never_seen):
        only_test_nums.append(num_list.pop(random.choice(list(range(2,len(num_list)-1)))))
    print(only_test_nums)
    print(len(num_list))
    # Generate subset of numbers for each number slot for the training set
    num1_train = [ num_list[i] for i in sorted(random.sample(range(len(num_list)), 70)) ]
    num2_train = [ num_list[i] for i in sorted(random.sample(range(len(num_list)), 70)) ]
    num3_train = [ num_list[i] for i in sorted(random.sample(range(len(num_list)), 70)) ]
    # for each number slot in the testing set add numbers 
    # that aren't present in that slot in training set
    num1_test = []
    num2_test = []
    num3_test = []
    for num in num_list:
        if num not in num1_train:
            num1_test.append(num)
        if num not in num2_train:
            num2_test.append(num)
        if num not in num3_train:
            num3_test.append(num)
    print(num1_test)
    print(num2_test)
    print(num3_test)
    #Generate traing set
    training_set = []
    #Generate one operand questions
    for op in ops:
        num1s = [ num1_train[i] for i in sorted(random.sample(range(len(num1_train)), 50)) ]
        for num1 in num1s:
            num2s = [ num2_train[i] for i in sorted(random.sample(range(len(num2_train)), 50)) ]
            for num2 in num2s:
                question  = str(num1) + op + str(num2)
                answer = str(eval(question))
                training_set.append([question,answer,[num1,num2,op]])
                #print('training ', question,' ',answer)
    # Generate two operand questions
    for op in ops:
        for op2 in ops:
            num1s = [ num1_train[i] for i in sorted(random.sample(range(len(num1_train)), 50)) ]
            for num1 in num1s:
                num2s = [ num2_train[i] for i in sorted(random.sample(range(len(num2_train)), 50)) ]
                for num2 in num2s:
                    num3s = [ num3_train[i] for i in sorted(random.sample(range(len(num3_train)), 50)) ]
                    for num3 in num3s:
                        question  = str(num1) + op + str(num2) + op2 + str(num3)
                        answer = str(eval(question))
                        training_set.append([question,answer,[num1,num2,num3,op,op2]])
                        #print('training ', question,' ',answer)
    # Generate testing set
    testing_set = []
    for op in ops:
        num1s = deepcopy(only_test_nums)
        num1s.extend(num1_test)
        print(num1s)
        #num1s.extend(num1_train)
        for num1 in num1s:
            num2s = deepcopy(only_test_nums)
            num2s.extend(num2_test)
            print(num2s)
            #num2s.extend(num2_train)
            for num2 in num2s:
                question  = str(num1) + op + str(num2)
                answer = str(eval(question))
                #if [question,answer,[num1,num2,op]] not in training_set:
                testing_set.append([question,answer,[num1,num2,op]])
                print('testing ', question,' ',answer)
    # Generate two operand questions
    for op in ops:
        for op2 in ops:
            num1s = deepcopy(only_test_nums)
            num1s.extend(num1_test)
            print(num1s)
            #num1s.extend(num1_train)
            for num1 in num1s:
                num2s = deepcopy(only_test_nums)
                num2s.extend(num2_test)
                print(num2s)
                #num2s.extend(num2_train)
                for num2 in num2s:
                    num3s = deepcopy(only_test_nums)
                    num3s.extend(num3_test)
                    print(num3s)
                    #num3s.extend(num3_train)
                    for num3 in num3s:
                        question  = str(num1) + op + str(num2) + op2 + str(num3)
                        answer = str(eval(question))
                        #if [question,answer,[num1,num2,num3,op,op2]] not in training_set:
                        testing_set.append([question,answer,[num1,num2,num3,op,op2]])
                        print('testing ', question,' ',answer)
    pickle.dump(training_set,open('training_easy_+-x.pkl','wb'))
    pickle.dump(testing_set,open('testing_easy_+-x.pkl','wb'))

def indexesFromExpression(expression,index_dict = sym2index):
    return [index_dict[sym] for sym in expression]


def tensorFromExpression(expression):
    indexes = indexesFromExpression(expression)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromExpression(pair[0])
    target_tensor = tensorFromExpression(pair[1])
    return (input_tensor, target_tensor)


