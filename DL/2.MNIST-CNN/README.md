# MNIST-CLASSIFICATION
## Dataset
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.
The MNIST database of handwritten digits, has a training set of 60,000 28x28 grayscale images of the 10 digits, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.<br/>
Sample images from MNIST test dataset:<br/>
![github-small](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
## Image classification CNN model on MNIST dataset
The model consists of 2 convolutional layers which are followed by maxpooling layers.The output of these layers is then flattened and fed into the dense layers which give the final output.
## Keras implementation accuracy
Train accuracy: 99.66%<br/>
Test accuracy: 99.12%<br/>

## Tensorflow implementation accuracy
Test accuracy: 98.55%<br/>
