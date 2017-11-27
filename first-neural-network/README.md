# Project #1 - Predicting Bike Sharing Data

In this project, I built a simple two-layer neural network to predict bike sharing usage.

### Summary

**Dataset:** See the *Bike-Sharing-Dataset* folder for more details!

**Important files:** 
- *Your_first_neural_network.ipynb* contains most of the assignment details and shows the training and results of the neural network.
- *my_answers.py* contains my own work on the project, as detailed below.
  - `forward_pass_train()` implements the forward pass of the neural network, primarily matrix multiplication of input features, weights, and activations.
  - This is a regression problem, so no activation function is used on the output layer.
  - `backpropagation()` implements the backpropagation through the neural network of the error between prediction and the label. This multiplies the error by related weights of each layer, along with the gradient of each layer.
  - The change in weights based on this error backpropagation is then calculated.
  - `update_weights()` just takes the calculated change in weights times the desired learning rate
  - This has also been modified by the constant division by the number of records in the related training batch
  - `run()` only performs a forward pass (i.e. non-training)
  
