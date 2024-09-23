import torch
from torch import nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)



class LinearRegression(nn.Module):
	def __init__(self):
		super().__init__()
		self.weight = nn.Parameter(data=torch.randn(1), requires_grad=True)
		self.bias = nn.Parameter(data = torch.randn(1), requires_grad=True)

		# self.weight = nn.Parameter(data=torch.Tensor([3]), requires_grad=True)
		# self.bias = nn.Parameter(data =torch.Tensor([2]), requires_grad=True)
		print("self.weight: ", self.weight.size())

	def forward(self, x):
		output = x@self.weight + self.bias

		return output




def train():
	for epoch in range(1, num_epochs+1):
		y_predict = model(X)
		print("y_predict: ", y_predict.size())
		print("y: ", y_predict.size())
		loss = loss_fn(y_predict, y)

		print(f"Epoch: {epoch} - loss: {loss}")
		# print(f"Parameters: {model.parameters()}")

		# because gradient of optimizer is accumudation, we must set to zero before computing gradient
		optimizer.zero_grad()


		# computes the gradient of the loss respect parameter
		loss.backward()

		# update parameters that defined requires_grad = True
		optimizer.step()

		# evaluate after each epoch




def eval(model, X):
	model.eval()
	with torch.inference_mode():
		y_predict = model(X)

		

if __name__ == "__main__":
	# 1. Creat dumy dataset

	# y = 3 * x + 2
	X = [x for x in range(1, 100)]
	y = [3*x + 2 for x in X]


	X = torch.Tensor(X)
	y = torch.Tensor(y)



	X = X.view(X.shape[-1], -1)
	y = y.view(X.shape[-1], -1)


	print(X.size())
	print(y.size())



	model = LinearRegression()

	loss_fn = nn.L1Loss()

	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

	num_epochs = 1000



	train()




