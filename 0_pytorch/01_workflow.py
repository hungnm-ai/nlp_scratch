import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from safetensors.torch import save_file, load_file



torch.manual_seed(42)
np.random.seed(42)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(data=torch.randn(1), requires_grad=True)

    def forward(self, x):
        output = x @ self.weight + self.bias
        return output


def count_number_parameters(model):
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    print(
        f"Total parameters: {total_parameters} - parameters trainable: {trainable_parameters}"
    )


def visualize_training_process(df):
    # Set the figure size for the plot
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="epoch", y="train_loss", label="Train loss")
    sns.lineplot(data=df, x="epoch", y="eval_loss", label="Eval loss")

    # Add labels and title
    plt.title("Training and Evaluation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def save_checkpoint(model, checkpoint_dir: str, save_safetensors: bool = True):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = "model.safetensors" if save_safetensors else "model.pth"
    model_path = os.path.join(checkpoint_dir, model_name)

    if save_safetensors:
    	save_file(model.state_dict(), model_path)
    else:

    	torch.save(model.state_dict(), model_path)

    print(f"Save checkpoint to: {model_path}")

def load_state_dict(checkpoint_dir: str):
	model_path = os.path.join(checkpoint_dir, "model.safetensors")
	if os.path.exists(model_path):
		state_dict = load_file(model_path)
	else:
		model_path = os.path.join(checkpoint_dir, "model.pth")
		if not os.path.exists(model_path):
			model_path = os.path.join(checkpoint_dir, "model.pt")
			if not os.path.exists(model_path):
				raise Exception("Could not load model!") 

		state_dict = torch.load(model_path)

	return state_dict




X = np.array([random.random() for _ in range(1, 1000)], dtype=np.float32)
y = 3 * X + 2  # assume that ground truth y = 3*X + 2


X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.from_numpy(X_train)
X_eval = torch.from_numpy(X_eval)

y_train = torch.from_numpy(y_train)
y_eval = torch.from_numpy(y_eval)


# Reshape X and y to (1000, 1), 1000 example and one feature
X_train = X_train.view(-1, 1)
X_eval = X_eval.view(-1, 1)


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

print(f"Model is training on {device}")

# model = LinearRegression()
# model = model.to(device)

# count_number_parameters(model)

# # MAE
# loss_fn = nn.L1Loss()

# lr = 0.05
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# num_epochs = 100

# report = []

# for epoch in range(1, num_epochs + 1):
#     ### Training
#     model.train()

#     # move data to device
#     X_train = X_train.to(device)
#     y_train = y_train.to(device)

#     # forward to create output (y_predict)
#     y_predict = model(X_train)

#     # compute loss
#     loss = loss_fn(y_predict, y_train)

#     # compute gradients
#     loss.backward()

#     # update parameters based on computed gradient
#     optimizer.step()

#     # reset gradients
#     optimizer.zero_grad()

#     model.eval()
#     ### Evaluate after each epoch
#     with torch.inference_mode():

#         X_eval = X_eval.to(device)
#         y_eval = y_eval.to(device)

#         y_eval_predict = model(X_eval)

#         eval_loss = loss_fn(y_eval_predict, y_eval)

#         loss = loss.to("cpu").item()
#         eval_loss = eval_loss.to("cpu").item()

#         print(f"Epoch: {epoch}: Train loss={loss} - Eval loss={eval_loss}")
#         report.append({"epoch": epoch, "train_loss": loss, "eval_loss": eval_loss})

#     # save checkpoint model
#     if epoch % 10 == 0:
#         save_checkpoint(model, f"./model/checkpoint-{epoch}")


# df = pd.DataFrame(report)
# visualize_training_process(df)

### Inference
model = LinearRegression()
state_dict = load_state_dict("./model/checkpoint-100")
print("state_dict: ", state_dict)
model.load_state_dict(state_dict)
x_test = torch.tensor([0.4], dtype=torch.float32)
print(x_test.size())

model.eval()
with torch.inference_mode():
	y_test_pred = model(x_test)
	print(y_test_pred)

