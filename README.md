# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[allclasses]: ./img/allclasses.png "All Classes"
[classhisto]: ./img/classhisto.png "Class Histogram"
[effects]: ./img/effects.png "Image Effects"
[websigns]: ./img/websigns.png "Web Signs"
[webtopk]: ./img/webtopk.png "Web Top 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nnadams/SDC-TrafficSigns-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is one example image for each of the 43 classes. Below are histograms of the classes to show that many classes have very few examples.

![alt text][allclasses]

![alt text][classhisto]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For the preprocessing the data in first normalized and then augmented. Each image is converted to grayscale, has its histogram equilized and the mean is zeroed. This is to save the network from having to care about contrast or brightness.

![alt text][effects]

After visualizing in the last step the dataset, it is clear that some classes have fewer examples than others. The images copied and are augmented at least once, but classes with fewer images have this step repeated multiple times. In the end each class of sign has at least 3000 examples. New images will have some subset of: translating, shearing, rotating, gamma changes, or blurring applied to them. These effects allow slightly different images to be seen by the network. Signs are in all kinds of locations in many different environments. See below for examples.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

This network is a slightly modified LeNet from class, including larger fully connected layers and dropout.

|    **Layer**    |               **Description**               |
| :-------------: | :-----------------------------------------: |
|      Input      |    32x32x1 preprocessed grayscale image     |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6  |
|      RELU       |                                             |
|   Max Pooling   | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                             |
|   Max Pooling   |  2x2 stride, valid padding, outputs 5x5x16  |
|     Dropout     |            80% Training Dropout             |
| Fully Connected |                 Outputs 512                 |
|      RELU       |                                             |
| Fully Connected |                 Outputs 84                  |
|      RELU       |                                             |
| Fully Connected |                 Outputs 43                  |
|     Output      |               43 Sign Classes               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is trained over 15 epochs in batches of 128 at a learning rate of 0.0005. The AdamOptimizer was chosen as a good default starting point. I left these mostly untouched during the development process. Though I found the increasing the learning rate higher gave issues with accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

```
Train Accuracy: 0.872876499243
Validation Accuracy: 0.966439909054
Test Accuracy: 0.939192398937
```

I elected to begin with the LeNet-5 architecture, because it is designed for small black and white images. I believed this would be a good starting point. The traffic sign images I used were only slightly bigger 32x32 vs 28x28 and grayscale rather than binary. At first I did try the full color images in LeNet with very poor results. Moving to grayscale and augmenting the images helped to get the accuary just around 75% for the training set. However the validation set remained lower. I added the dropout layer to help with the overfitting. I did try a handful of other options at this point, such as different activation functions, more convolutional layer, and more dropout layers. However those did not improve accuracy.

Original LeNet worked well with much simpler digit images, and going from 10 digits to 43 signs clearly required more parameters. However the images on the signs aren't very complex. So I believed the convolutional layers would be able to stay that same and instead increased the size of the first fully connected layer. This worked to greatly improve my testing and validation accuracies.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][websigns]

I chose some signs from photos that were old looking or vandalized. Unfortunatly that doesn't come through in 32x32 very well. I also added 3 stop signs with blinders. Normally these are used when two lanes are merging and allow just one side to see the sign. I was curious to see what the model would classify them as having never seen them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

As you can see below, the model was able to correctly guess the 5 images but was tripped up by the stop signs with blinders. Looking head on it was correct, but looking from either side it was not able to, probably beause of shadows.

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 70 kph                | 70 kph                                        |
| Stop     			    | Stop                                          |
| No Entry				| No entry										|
| Road Work	      		| Road work	    				 				|
| Ahead only			| Ahead only          							|
| Stop (blinders) | Keep Left                                     |
| Stop (blinders) | Stop                                          |
| Stop (blinders) | Stop                                        |


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 88%. This compares favorably to the accuracy on the test set of 93%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Overall the model on average is certain about the correct predictions it is giving. Also notably it is finding patterns, because you can see the highest probabilities for the triangle shaped signs are almost triangles, similarly for the circle shapes. The stop signs with blinders as expected have lower probabilities.

Below are the top 5 softmax probabilities for each web image.

![alt text][webtopk]

