import torch
import torch.nn.functional as F
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn, optim

### 1. Prepare dataset

# create dumy dataset
n_samples = 1000

# Create dataset has shape like circle
X, y = make_circles(n_samples, noise=0.001, random_state=42)

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
		self.linear1 = nn.Linear(in_features=2, out_features=16, bias=True)
		# self.linear2 = nn.Linear(in_features=8, out_features=2, bias=True)
		self.linear3 = nn.Linear(in_features=16, out_features=1, bias=True)



	def forward(self, x):
		output1 = self.linear1(x)
		# output2 = self.linear2(output1)
		output = self.linear3(output1)

		return torch.sigmoid(output)

def metrics(y_true, y_pred):
    # Round the output probabilities to get binary predictions (0 or 1)
    y_pred_class = torch.round(y_pred)
    correct = torch.eq(y_true, y_pred_class).sum().item()  # Compare the true labels with predicted labels
    acc = (correct / len(y_pred)) * 100
    return acc


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
optimizer = optim.SGD(params=model.parameters(), lr=0.5)


num_epochs = 100

X_train = X_train.to(device)
y_train = y_train.to(device)
for epoch in range(1, num_epochs + 1):
	model.train()

	
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
	with torch.no_grad():

		X_eval = X_eval.to(device)
		y_eval = y_eval.to(device)
		y_eval_predict = model(X_eval)
		y_eval_predict = y_eval_predict.squeeze()
		eval_loss = loss_fn(y_eval_predict, y_eval)

		train_acc = metrics(y_train, y_train_predict)
		eval_acc = metrics(y_eval, y_eval_predict)

		print(f"Epoch {epoch} | Training loss: {loss:.4f} | Eval loss: {eval_loss:.4f} | Train Acc: {train_acc:.2f}% | Eval Acc: {eval_acc:.2f}%")










