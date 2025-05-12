## 01 - Image as data

**Summary:** In this lesson, we'll learn how to work with tensors in PyTorch. We'll also explore the dataset for this project, focusing on how images are represented in tensors.

**Objectives:**
* Check important attributes of tensors, such as size, data type, and device.
* Manipulate tensors through slicing.
* Perform mathematical operations with tensors, including matrix multiplication and aggregation calculations.
* Download and decompress the dataset for this project.
* Load and explore images using PIL.
* Demonstrate how visual information is stored in tensors, focusing on color channels.

**New Terms:**
* Attribute
* Class
* Color channel
* Method
* Tensor

Link: [Conser-vision Practice Area: Image Classification](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/page/483/)

## 02 - What Are Python Tracebacks?

In Python, a traceback is a detailed report generated when an error occurs in your code. A traceback, also called a stack trace, offers valuable information about what and where went wrong in the code by providing a step-by-step account of what lead up to the Python raising an exception. While tracebacks may appear daunting at first glance, they contain crucial details that can significantly aid in debugging your code.

By carefully examining a traceback, you can:

* Understand the nature of the exception
* See the sequence of code that led to the error
* Identify the exact line where the error occurred, sometimes even where in the line the error occurred

Learning to read and understand tracebacks is an essential skill for Python developers because it's one of the primary ways to debug errors in Python code.

## 03 - What Is a Neural Network?
A neural network is a computational model inspired by the human brain's structure. It consists of layers of interconnected nodes, or "neurons," that process input data to recognize patterns and make decisions.​

**How It Works**

*Inputs and Outputs:* The network receives input data (e.g., pixel values from an image) and processes it through multiple layers to produce an output (e.g., identifying a digit).​

*Weights and Biases:* Each connection between neurons has an associated weight, determining the influence of one neuron on another. Biases are additional parameters that adjust the output along with the weighted sum.​

*Activation Functions:* After computing the weighted sum and adding the bias, an activation function (like the sigmoid or ReLU) is applied to introduce non-linearity, enabling the network to model complex patterns.​

*Training Through Backpropagation:* The network learns by comparing its output to the actual result, calculating the error, and adjusting the weights and biases to minimize this error. This process is repeated over many iterations using algorithms like gradient descent.
[What is a neural network? | Deep learning chapter 1](https://youtu.be/aircAruvnKk?si=BW88KYVbsFZhXPGW)

## Gradient Descent, How Neural Networks Learn

*Cost Function:* Measures the difference between the network's predictions and actual results. The goal is to minimize this error.​

*Gradient Descent:* An optimization technique where the network adjusts its weights in the direction that most reduces the cost function, akin to descending a slope to reach the lowest point.​

*Training Process:* The network uses labeled data to iteratively update its weights, improving its predictions over time.
[Gradient descent, how neural networks learn](https://youtu.be/IHZwWFHWa-w)

## Exercise - Binary Classification with PyTorch

**Summary:** In this lesson, we'll continue using PyTorch and build our very first neural network model. We'll use this model to classify if a wildlife camera image shows a hog or not.

**Objectives:**
* Convert images from grayscale to RGB
* Resizes images
* Create a transformation pipeline to * standardize images for training
* Build and train a simple neural network model in PyTorch
* Save our trained neural network to disk

**New Terms:**
* Activation function
* Automatic differentiation
* Backpropagation
* Binary classification
* Cross-entropy
* Epoch
* Layers
* Logits
* Optimizer

## 04 - What is Backpropagation really doing?
Backpropagation is a learning algorithm used in neural networks to improve accuracy. It works by:

**Calculating the error** – It compares the network’s prediction to the actual result.

**Adjusting weights and biases** – It updates the weights (which control the strength of connections between neurons) and biases (which shift the activation) to reduce the error.

**Repeating the process** – This adjustment is done repeatedly over many training examples, allowing the network to learn over time.

In simple terms, backpropagation helps the neural network learn by correcting its mistakes step by step.

---
#### Key Concepts
**Error Measurement:** The network calculates the difference between its predictions and the actual outcomes using a cost function.​

**Gradient Calculation:** Backpropagation computes the gradient of the cost function with respect to each weight and bias, determining how changes in these parameters affect the overall error.​
3Blue1Brown

**Parameter Update:** Using the gradients, the network updates its weights and biases in the direction that most reduces the error, effectively "learning" from its mistakes. [Backpropagation, intuitively](https://youtu.be/Ilg3gGewQ5U?si=22Y3ieLURD2fLG6F)

## What Is a Convolutional Neural Network?
A CNN is a specialized type of neural network designed to process and analyze visual data, such as images and videos. Unlike traditional neural networks, CNNs are particularly effective for tasks like image classification, object detection, and facial recognition.​

**Key Components**\
*Convolutional Layers:* These layers apply filters (also known as kernels) to the input image to detect features like edges, textures, and patterns.​
Medium

*Activation Functions:* After convolution, activation functions like ReLU (Rectified Linear Unit) introduce non-linearity, enabling the network to learn complex patterns.​

*Pooling Layers:* Pooling reduces the spatial dimensions of the feature maps, decreasing computational load and helping prevent overfitting.​

*Fully Connected Layers:* These layers connect every neuron to every other neuron, facilitating the final decision-making process, such as classifying an image.​

**Why CNNs Are Effective -**
CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images. This means they can learn to recognize complex patterns by combining simpler ones, making them highly efficient for visual tasks.
[Convolutional Neural Network](https://youtu.be/zfiSAzpy9NM)

## Exercise - Multiclass Classification
**Summary:** In this lesson, we'll work with the full wildlife dataset, which has eight classes. This is more than the network in the previous notebook can handle. Here we'll build and train a more complicated neural network, called a Convolutional Neural Network, that is meant for working with images. We'll use this network to get the predictions we need for the competition at DrivenData.org.

**Objectives:**
* Read in data with multiple classes
* Normalize our data to improve performance
* Create a Convolutional Neural Network that works well with images
* Train that network to do multiclass classification
* Reformat the network predictions to complete the competition

**New Terms:**
* Multiclass
* Normalize
* Convolution
* Max pooling