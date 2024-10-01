import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim


### 1. Prepare dataset

# create dumy dataset
n_samples = 1000

# Create dataset has shape like circle
X, y = make_circles(n_samples, noise=0.03, random_state=42)

print(f"types of X: {type(X)} - shape: {X.shape}")
print(f"types of y: {type(y)} - shape: {y.shape}")

# Convert numpy data to tensor
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


print(f"types of X: {type(X)} - shape: {X.shape}")
print(f"types of y: {type(y)} - shape: {y.shape}")

# split data into train - dev - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(f"shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"shape of X_eval: {X_eval.shape}, y_eval: {y_eval.shape}")
print(f"shape of X_test: {X_test.shape}, y_test: {y_test.shape}")


### 2. Define architecture model 

class BaseCls(nn.Module):
	"""
	A simple feedforward neural netword with 3 fully connected layers.

	"""
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(in_features=2, out_features=4, bias=True)
		self.linear2 = nn.Linear(in_features=4, out_features=2, bias=True)
		self.linear3 = nn.Linear(in_features=2, out_features=1, bias=True)
		self.sigmoid = nn.Sigmoid()



	def forward(self, x):
		output1 = self.linear1(x)
		output2 = self.linear2(output1)
		output = self.linear3(output2)

		return self.sigmoid(output)

def metrics(predicts, labels):
	pass


### 3. Training model

device = "cpu"
if torch.backends.mps.is_available():
	device = "mps"
if torch.cuda.is_available():
	device = "cuda"

# define model
model = BaseCls()
model = model.to(device)

# define loss function
loss_fn = nn.BCELoss()

# define optimizer
optimizer = optim.SGD(params=model.parameters(), lr=0.01)


num_epochs = 20


for epoch in range(1, num_epochs + 1):
	model.train()

	X_train = X_train.to(device)
	y_train = y_train.to(device)
	y_train_predict = model(X_train)
	y_train_predict = y_train_predict.squeeze()
	loss = loss_fn(y_train_predict, y_train)



	# compute gradient descent
	loss.backward()

	# update weights
	optimizer.step()

	# reset gradient
	optimizer.zero_grad()


	# evaludate
	model.eval()
	with torch.inference_mode():

		X_eval = X_eval.to(device)
		y_eval = y_eval.to(device)
		y_eval_predict = model(X_eval)
		y_eval_predict = y_eval_predict.squeeze()
		print(y_eval_predict)
		eval_loss = loss_fn(y_eval_predict, y_eval)

	print(f"Epoch {epoch} | Training loss: {loss} - Eval loss: {eval_loss}")









