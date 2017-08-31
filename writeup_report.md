#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_visualization]: ./writeup_report_files/model_visualization.png "Model Visualization"
[recovery_left]: ./writeup_report_files/recovery_left.png "Recovery Image"
[recovery_right]: ./writeup_report_files/recovery_right.png "Recovery Image"
[placeholder_small]: ./writeup_report_files/center.png "Center Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (unmodified)
* model.h5 containing a trained convolution neural network 
* video.mp4 autonomous mode video of one round of track 1 (recorded using model.h5)
* writeup_report.md (this document) summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses NVidia's [DAVE-2](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025) network architecture. It consists of convolutional neural network (see the create_model() function) with:

- A normalization layer
- A cropping layer to remove the sky and hood of the car
- 5 convolutional layers, each having half the resolution of the previous layer but increasingly larger filter-sets
- A flattening layer
- 4 dense/fully connected layers of increasingly smaller size perform classification, the last single-node layer outputs the steering angle


####2. Attempts to reduce overfitting in the model

The model does not contain dropout layers in order to reduce overfitting. I tried adding dropout layers between the dense layers, before the network wasn't performing well enough, but dropout seemed to only make things worse. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track number 1. To my surprise, while it was only trained on track 1, it managed to complete much of track 2 before it drove off a cliff, it got confused by a sharp curve without any clear separation of the road at the start and end of the curve and made a fatal attempt to take a shortcut.


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py, train_model() function).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, taking sharp curves near the inside and reversing direction (outside of recording sessions!) to reduce left/right bias. Besides the center image, both the left and right camera images were used. The left/right images were required to make the car steer properly in curves, a small steering angle correction of +/- 0.3 was made to stay on track.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
