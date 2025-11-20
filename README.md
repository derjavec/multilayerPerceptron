# multilayerPerceptron

A clean-room implementation of a **Multi-Layer Perceptron (MLP)** from scratch, focused on understanding how neural networks work internally: initialization, forward propagation, backpropagation, and optimization.

---

## 📌 What is a Multi-Layer Perceptron?

The **Multi-Layer Perceptron** is one of the simplest neural network architectures.
Its power depends entirely on its **configuration parameters** and the **amount/quality of data**.

An MLP is composed of:

* An **input layer** (receives raw features)
* One or more **hidden layers** (learn hierarchical representations)
* An **output layer** (produces class probabilities using Softmax)

Each neuron is a perceptron that learns a weighted combination of its inputs, applies an activation function, and passes the result forward.

---

## 🧠 Network Structure

Let:

* Input: **X ∈ ℝ(m × n_features)**
* Hidden layers: **X ∈ ℝ(m × n_neurons_prev)**
* Output: **X ∈ ℝ(m × n_classes)**

Where **m** is the number of samples.

The final layer uses **Softmax**, producing a probability distribution over classes for each sample.

---

## ⚙️ Configuration Parameters

The model is controlled by a config file specifying:

* Layer sizes (neurons per layer)
* Learning rate
* Number of epochs
* Batch size
* Activation function per layer

These parameters directly affect training speed, stability, and final performance.

---

## 🚀 Training Process Overview

Training consists of iterating through:

1. **Initialization**
2. **Feature scaling**
3. **Epoch loop**
4. **Batch loop**
5. **Forward propagation**
6. **Backward propagation (Gradient Descent)**
7. **Loss & Accuracy calculation**
8. **Model saving**

---

## 1️⃣ Weight Initialization

Weights are initialized randomly with small values using:

### ✅ Xavier (Glorot) Initialization

Used for: **Sigmoid / Tanh**
[
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, ; \sqrt{\frac{6}{n_{in} + n_{out}}} \right)
]

or
[
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
]

### ✅ He Initialization

Used for: **ReLU**
[
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
]

Where:

* (n_{in}) = number of input neurons
* (n_{out}) = number of output neurons

Biases are usually initialized as zeros.

---

## 2️⃣ Feature Scaling

All features must be normalized or standardized:
[
X_{scaled} = \frac{X - \mu}{\sigma}
]

This improves stability and convergence speed.

---

## 3️⃣ Forward Propagation

For each layer (l):

[
Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}
]
[
A^{(l)} = f(Z^{(l)})
]

Where:

* (f) is the activation function
* (A^{(0)} = X)

Every (Z), (A) and derivative is stored for backpropagation.

---

## 4️⃣ Activation Functions

### 🔵 ReLU

[
f(x) = \max(0, x)
]
[
f'(x) = \begin{cases}
1 & x > 0 \
0 & x \leq 0
\end{cases}
]

Good for hidden layers due to reduced vanishing gradient problem.

---

### 🔵 Sigmoid

[
f(x) = \frac{1}{1 + e^{-x}}
]
[
f'(x) = f(x)(1 - f(x))
]

Used mainly for simple models. Can suffer from vanishing gradients.

---

### 🔵 Softmax (Output Layer)

[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
]

Outputs a probability distribution over (C) classes.

---

## 5️⃣ Backpropagation & Gradient Descent

For each batch, error is propagated backward:

### Gradient Descent Update Rule

[
W = W - \eta \cdot \frac{\partial L}{\partial W}
]
[
b = b - \eta \cdot \frac{\partial L}{\partial b}
]

Where:

* (\eta) = learning rate
* (L) = loss function

Backpropagation computes derivatives from output layer to input layer using the chain rule.

---

## 6️⃣ Loss & Accuracy

### ✅ Cross-Entropy Loss (Softmax)

[
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
]

Where:

* (y) = true labels (one-hot encoded)
* (\hat{y}) = predicted probabilities

---

### ✅ Accuracy

[
Accuracy = \frac{Number\ of\ correct\ predictions}{Total\ predictions}
]

Equivalent form:
[
Accuracy = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}(argmax(\hat{y}_i) = argmax(y_i))
]

---

## 7️⃣ Training Loop Structure

Pseudo-flow:

```
for epoch in range(epochs):
    for each batch:
        forward()
        backward()
        update_weights()

    compute_loss()
    compute_accuracy()
```

During training, weights are updated at every batch using gradient descent.

---

## 8️⃣ Model Saving

After training, the model is stored with:

* Configuration
* Learned weights
* Biases

This allows reloading and inference without retraining.

---

## 🔮 Prediction

To predict:

1. Load model + weights
2. Apply same feature scaling
3. Forward propagate through the network
4. Get class probabilities via Softmax

---

## 🎯 Goal

The goal of this MLP is to learn complex patterns between features and target labels, even when no linear relationship exists.

If the configuration is incorrect, the model may:

* Learn very slowly
* Fail to converge
* Explode or vanish gradients

Careful tuning is essential.

---

## 📎 Notes

This project is purely educational and designed to understand MLP mechanics without deep learning libraries like TensorFlow or PyTorch.

    