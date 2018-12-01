from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import nn_modules
import dataset_generator
import data_modifiers

def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		target_classes = data_modifiers.torch_vec_2_classlabel(target) #need this right now 
		# Torch doesn't support soft labels!!!!!!
		#print(target_classes.shape)
		#print(target_classes)
		#print(data.shape)
		#print(data)
		#print(target.shape)
		#print(target)
		optimizer.zero_grad()
		output = model(data)
		#print(output.shape)
		#print(output.shape)
		#print(output)
		loss = F.nll_loss(output, target_classes)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			target_classes = data_modifiers.torch_vec_2_classlabel(target) #need this right now 
			# Torch doesn't support soft labels!!!!!!
			output = model(data)
			test_loss += F.nll_loss(output, target_classes, size_average=False).item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target_classes.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='Arithmetic Net')
	parser.add_argument('--data-file', type=str, default='/hd1/scsnl/neural_networks/modular_math_cog/exp1/data/datasets/2000_+-_0.5.pkl', metavar='N',
						help='data file to use (default: /hd1/scsnl/neural_networks/modular_math_cog/exp1/data/datasets/2000_+-_0.5.pkl)')	
	parser.add_argument('--batch-size', type=int, default=100, metavar='N',
						help='input batch size for training (default: 100)')
	parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
						help='input batch size for testing (default: 100)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=4, metavar='N',
						help='how many batches to wait before logging training status')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = torch.utils.data.DataLoader(
		dataset_generator.three_num_abstract(args.data_file),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		dataset_generator.three_num_abstract(args.data_file,train = False),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)


	data_info = dataset_generator.three_num_abstract(args.data_file).get_info()
	print(data_info)

	model = nn_modules.Basic_ANN(data_info['input_dim'],200,data_info['label_dim']).to(device)
	#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader)


if __name__ == '__main__':
	main()