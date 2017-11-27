# Project #1 - Predicting Bike Sharing Data

In this project, I built a simple two-layer neural network to predict bike sharing usage.

### Summary

**Dataset:** See the *Bike-Sharing-Dataset* folder for more details!

**Important files:**

*Your_first_neural_network.ipynb* contains most of the assignment details and shows the training and results of the neural network.

*my_answers.py* contains my own work on the project, as detailed below.
- `forward_pass_train()` implements the forward pass of the neural network, primarily matrix multiplication of input features, weights, and activations.
- This is a regression problem, so no activation function is used on the output layer.
- `backpropagation()` implements the backpropagation through the neural network of the error between prediction and the label. This multiplies the error by related weights of each layer, along with the gradient of each layer.
- The change in weights based on this error backpropagation is then calculated.
- `update_weights()` just takes the calculated change in weights times the desired learning rate
- This has also been modified by the constant division by the number of records in the related training batch
- `run()` only performs a forward pass (i.e. non-training)
- This also contains certain hyperparameters to tune the neural network:
  - `iterations`, for the number of times to run through training rounds with the dataset
  - `learning_rate` - how quickly to update the weights after each training iteration (see above regarding the **0.5** I selected being smaller due to division by number of training examples)
  - `hidden_nodes` for the number of nodes in the hidden layer. Smaller learning rates needed a higher number of iterations to reach the desired loss, while higher number of hidden_nodes also needed more iterations or struggled to reduce loss further. 

Although `output_nodes` is listed as a hyperparameter, this can only be one for this project, as it is a regression problem for a single value (predicted bike sharers in a given hour on a given day).
