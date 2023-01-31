<!-- # convexMTLPyTorch
Implementation of ConvexMTL neural networks using PyTorch. -->

# ConvexMTLPyTorch

It includes a PyTorch lightning module, called ConvexTorchCombinator, for multi-task learning with a convex combination of task-specific and common models. It is implemented using the PyTorch Lightning module.

Mathematically, the model $h_r(\cdot)$ for each task can be defined as
$ h_r(x) = \lambda_r g(x) + (1-\lambda_r) g_r(x), $
with $\lambda_r \in [0, 1]$. Here $g(\cdot)$ is a common module and $g_r(\cdot)$ a task-specific one. 

With this implementation, any PyTorch modules can be combined to build multi-task models.


This is the implementation used in the paper [Convex Multi-Task Learning with Neural Networks](https://www.scinapse.io/papers/4295135783).

## Installation

You can install convexMTLPyTorch using pip:
```bash
pip install convexMTLPyTorch
```

## Usage

Here is a simple example to get you started:

```python
import ConvexTorchCombinator

# Define tasks 
n_tasks = 3
tasks = range(n_tasks)

# Define common module
common_mod = NeuralNetwork

# Define specific modules
spec_mod = {t: NeuralNetork for t in tasks}

# Initialize the model
model = ConvexTorchCombinator(n_features=10, n_output=1, n_channel=1)

# Train the model
X_data = torch.randn(1000, self.n_features).double()
y = X_data[:, 0] ** 2 + torch.sin(X_data[:, 1]) - 2 * X_data[:, 2]
X_task = torch.randint(0, n_tasks, (1000, 1))
train_ds = TensorDataset(X_data, X_task, y)
train_dl = DataLoader(train_ds)

trainer = Trainer(max_epochs=100)
trainer.fit(model, train_dl)

# Make predictions
predictions = model(X_data, X_task)
```


