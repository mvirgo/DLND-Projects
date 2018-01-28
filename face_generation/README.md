# Project #4 - Generating Face Images

In this project, I built a generative adversarial network (GAN) capable of generating new images of human faces. 

### Summary

#### Dataset

There are two datasets used in this project:
1. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/), used for testing the GAN architecture on simpler data to check that training works appropriately.
2. The [CelebFaces Attributes dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), used as the real images fed to the discriminator, which the generator learns to mimic in order to fool the discriminator.

#### Project Steps

All the work done in this project is in the [dlnd_face_generation.ipynb](face_generation/dlnd_face_generation.ipynb) jupyter notebook.

1. Udacity already took care of the pre-processing of the images, which normalize the pixel values between -0.5 and 0.5, as well as cropping the CelebFaces images and re-sizing them to 28x28 images.
2. I built the discriminator for the network, which takes in the real images from the dataset or the fake images from the generator as input, and tries to determine whether the image is real or fake.
    - I used three convolutional layers with leaky ReLU, along with batch normalization on the second and third of these (leaky ReLU *follows* batch norm), before a final fully connected layer leading to the output, with sigmoid activation.
3. I then built the generator for the network, which takes in a random noise vector and outputs a 28x28 image, which it attempts to fool the discriminator into believing as a real image.
    - From the input noise vector, there is a fully connected layer that is reshaped to be used with successive tranpose convolutional layers.
    - There are three transpose convolutional layers, the first two of which use batch normalization, followed by leaky ReLU. The output layer uses a tanh activation function.
4. Next up was defining the loss function, which has a loss made up of real and fake image losses for the discriminator, and fake image loss for the generator (i.e., it wants to minimize the loss from the discriminator determining the fake images are fake).
5. Set the optimization functions with Adam Optimizer.
6. Make a training function with TensorFlow.
7. Training the neural network for both networks. You can see examples of the generated images within the jupyter notebook.
    - On MNIST, there were clearly numbers being generated at the conclusion of training, although sometimes fuzzy or a little ambiguous.
    - For the faces, there were clearly faces being generated, if a little blurry and non-descript.
