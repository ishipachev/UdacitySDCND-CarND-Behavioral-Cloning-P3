# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[center1]: ./writeup_pics/center_1.jpg "Center training"
[center2]: ./writeup_pics/center_2.jpg "Center training"
[recover1a]: ./writeup_pics/recover1a.jpg "Recover 1"
[recover1b]: ./writeup_pics/recover1b.jpg "Recover 1"
[recover2a]: ./writeup_pics/recover2a.jpg "Recover 2"
[recover2b]: ./writeup_pics/recover2b.jpg "Recover 2"
[problem_bridge]: ./writeup_pics/problem_bridge.jpg "Problem bridge"
[problem_corner1]: ./writeup_pics/problem_corner1.jpg "Problem corner"
[problem_corner2]: ./writeup_pics/problem_corner2.jpg "Problem corner"
[orig]: ./writeup_pics/orig.png "Original image"
[flip]: ./writeup_pics/flip.png "Flipped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model implements simplified nvidia network without one fully connected layer but with addition dropout levels for regularisation purposes.
Nvidia network architecture can be found here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
In short, model consists of 5 convolutions layers. 3 of them followed by pooling layer and all 5 convolution layers followed by ReLu layer activation. Right after there is flatten layer followed by fully connected layers with a couple dropout layers.

Input data normalized by Lambda layer and cropped by Cropping2D layer to reduce number of inputs of neural network.

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 65 and 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 73). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a 2 full circle records with different directions (`try1` and `try2` folders). In order to stabilize solution near to lanes I added records with vehicle recovering it's path to center of the road from border of the road (`try3_restore` folder). I also found a couple complicated tricky places on track (bridge and some corners) where vehicle struggle to recognise correct direction. So I added records for these paths too (`try4` and `try5` folder).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get networks already working! Like nvidia network. And able to learn on my PC, not some huge cluster in data center.

My first step was to make sure network atleast works, so I wrote simple_network() function just to make sure all pipeline with simulator works fine. And it was.

Right after this I copied nvidia network and try it. Due huge amount of fully connected layers (a lot of weights) I was trying to reduce them. I remove almost all layers and start setting them back until I found working network.

Without any additional experiments I added a couple Dropout layers just in hope it will fits fine and helps me to dodge overfitting.

I've played a bit with Dropout numbers and stopped at 0.3 and 0.3 probability to drop weight. So in total I will have 0.49 weights left without any drops.

I was checking the run of simulator almost after every network changes, so to the final part I had vehicle run smooth on track with a couple tricky places: bridge and some ground border. To improve driving behavior in these cases, I just gathered more data in these places with vehicle restore it path when it close to border and just ordinary passing by. It works.

For more data for training I implemented simple data augmentation with flipping image and changing sign of the angle. So this procedure allows me to increase amount of picture two times without any additional training by bare hangs.

#### 2. Final Model Architecture

The final model architecture (model.py lines 56-69) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Lambda         		| 160x320x3 Input RGB image   					| 
| Cropping2D         	| crop with params (70, 25) for height	        |
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 3х3 convolution, ReLu activation	            |
| Convolution2D			| 3х3 convolution, ReLu activation	            |
| Flatten				|      									        |
| Dropout		        | drop a weight with 0.3 probability			|
| Fully connected		| outputs 100									|
| Dropout               |                                               |
| Fully connected		| outputs 10									|
| Fully connected		| outputs 1										|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center1]
![alt text][center2]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back on track. These images show what a recovery looks like after some skipped frames:

![alt text][recover1a]
![alt text][recover1b]

And another one:

![alt text][recover2a]
![alt text][recover2b]


To augment the data sat, I also flipped images and angles thinking that this would increase amount training data. For example, here is an original image and the same image that has been flipped:

![alt text][orig]
![alt text][flip]

Also I gathered places where my model struggled to find the right wheel angle:


![alt text][problem_bridge]
![alt text][problem_corner1]
![alt text][problem_corner2]


After the collection process, I had 15588 number of pictures to process, including augmented ones. 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation loss dynamics and simulation tests. I used an adam optimizer so that manually training the learning rate wasn't necessary.
