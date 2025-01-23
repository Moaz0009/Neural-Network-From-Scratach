# Neural Network From Scratch

This repository contains a pure NumPy implementation of a Neural Network, built from the ground up to provide a deep understanding of neural network mechanics. It includes forward and backward propagation, activation functions, cost functions, and error handling, all designed with clarity and efficiency in mind.

## Features

- **Activation Functions**: Linear, ReLU, Sigmoid, Tanh, Softmax.
- **Cost Functions**: Mean Squared Error (MSE), Binary Cross-Entropy, Categorical Cross-Entropy.
- **Forward and Backward Propagation**: Complete implementations for training and evaluation.
- **Metrics**: Built-in accuracy metric for performance evaluation.
- **Verbose Mode**: Option to print cost at each iteration for debugging and progress tracking.
- **Error Handling**: Comprehensive checks for input/output shapes, activation function validity, and cost function names.

## Getting Started

### Prerequisites

Make sure you have Python installed with the following dependencies:

- `numpy`

Install NumPy via pip if you don't already have it:

```bash
pip install numpy
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch
```

### Usage

1. Import the `NeuralNetwork` class from the provided script.
2. Define your dataset (inputs and outputs).
3. Configure the network with your desired architecture, activation functions, and cost function.
4. Train the network and evaluate its performance.

Example:

```python
import numpy as np
from neural_network import NeuralNetwork

# Sample dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize the neural network
nn = NeuralNetwork(
    layers=[2, 4, 1],
    activation_functions=["relu", "sigmoid"],
    cost_function="binary_crossentropy",
    verbose=True
)

# Train the network
nn.train(X, y, epochs=1000, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X)
print("Predictions:", predictions)
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

This project was developed as a learning tool to demystify the inner workings of neural networks. Inspired by various online tutorials and textbooks on deep learning.

---

Would you like to include specific sections, such as a troubleshooting guide or advanced configuration options? Let me know if you'd like me to adjust this!# Neural-Network-From-Scratach
