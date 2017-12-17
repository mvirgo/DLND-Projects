# Project #2 - Classifying Dog Breeds

In this project, I worked on various methods of building a convolutional neural network that can classify dog breeds. 

### Summary

#### Datasets

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `dog-project/dogImages`. 

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `dog-project/lfw`.

3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `dog-project/bottleneck_features`.

4. Download the [InceptionV3 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) for the dog dataset - note that this was the specific model I used for transfer learning on my final architecture.  Place it in the repo, at location `dog-project/bottleneck_features`.

#### Project Steps

1. Build a human face detector (given), and assess its accuracy on human and dog pictures.
2. Build a dog detector (given), and assess its accuracy on human and dog pictures.
3. Create a convolutional neural network (CNN) from scratch to classify dog breeds among 133 classes.
4. Use transfer learning (I used the InceptionV3 model pre-trained on ImageNet) to create a substantially more accurate classifier for the 133 dog breeds.
5. Create an algorithm with the CNN classifier to detect the breed on a picture input. For dogs, the predicted breed is returned, while for humans they get the breed they most look like. For any other animals, an error message is returned.
6. Test out the dog classifier on dog, human, and other pictures.
7. (Bonus) I modified the algorithm in order to allow it to output the top two breeds when the algorithm is less than 70% confident, meaning that any mutts will get their top two most likely breeds (humans typically get this as well as the algorithm is not confident for them on any given breed). If it is a pure-bred dog, it still returns just the single breed class.

#### Image References
The images in [`/test_images`](/dog-project/test_images) came from the below sources:
- Alec Baldwin - in the human dataset above
- Golden Dox - http://www.101dogbreeds.com/wp-content/uploads/2017/09/Golden-Dox-Images.jpg
- Cocker Spaniel - http://r.ddmcdn.com/s_f/o_1/APL/uploads/2014/10/adopting-a-cocker-spaniel0.jpg
- Golden Retriever - http://cdn1-www.dogtime.com/assets/uploads/gallery/golden-retriever-dogs-and-puppies/golden-retriever-dogs-puppies-7.jpg
- Horse - https://images2.onionstatic.com/clickhole/3564/7/original/600.jpg
- Labradoodle - http://cdn1-www.dogtime.com/assets/uploads/gallery/labradoodle-dog-breed-pictures/side-1.jpg
- Me - That's me!
- Raccoon - http://s7d2.scene7.com/is/image/woodstream/hh-us-lc-animals-raccoon-facts-1
