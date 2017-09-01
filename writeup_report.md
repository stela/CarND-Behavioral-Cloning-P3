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

[model_visualization]: ./writeup_report_files/cnn-architecture.png "Model Visualization"
[left]: ./writeup_report_files/left.jpg "Left Image"
[right]: ./writeup_report_files/right.jpg "Right Image"
[center]: ./writeup_report_files/center.jpg "Center Image"

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

My model uses NVIDIA's [DAVE-2](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025) network architecture. It consists of a convolutional neural network (see the create_model() function) with:

- A normalization layer
- A cropping layer to remove the sky and hood of the car
- 5 convolutional layers, each having half the resolution of the previous layer but increasingly larger filter-sets
- A flattening layer
- 4 dense/fully connected layers of increasingly smaller size perform classification, the last single-node layer outputs the steering angle


####2. Attempts to reduce overfitting in the model

The model does not contain dropout layers in order to reduce overfitting. I tried adding dropout layers between the dense layers, before the network was performing well enough, but dropout seemed to only make things worse.

The model was trained and validated on different data sets to ensure that the model was not overfitting (see training/validation split logic in load_train_and_validation_csv_lines()). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track number 1. To my surprise, while it was only trained on track 1, it managed to complete much of track 2 before it drove off a cliff, it got confused by a sharp curve without any clear separation of the road at the start and end of the curve and made a fatal attempt to take a shortcut.


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py, train_model() function).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, taking sharp curves near the inside and reversing direction (outside of recording sessions!) to reduce left/right bias. Besides the center image, both the left and right camera images were used. The left/right images were required to make the car steer properly in curves, a small steering angle correction of +/- 0.3 was made to stay on track.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deciding on a model architecture to start with, was partly determined by me having an old slow laptop to work on at the start of this project (limited CPU power and memory), and a desire to start with the most simple thing which could possibly work, in order to not make things unnecessarily complex. The NVIDIA "DAVE-2" model fit the bill perfectly, it had both a fairly simple structure and limited size, and it was created exactly for the task at hand, so I thought there was a good chance at success with it.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (see load_train_and_validation_csv_lines()). After adding support for left/center/right images (with steering-angle compensation), the model had a similar mean squared error on the training set and on the validation set, implying there is little or no overfitting. That probably explains why my attempts of using dropout didn't improve driving performance. I think the lack of overfitting is also thanks to using a just-large-enough network, some of the networks introduced in the transfer learning and behaviour cloning section had a huge number of layers and nodes which might be more suited for more complex or abstract tasks than recognizing road and non-road.

The final step was to run the simulator to see how well the car was driving around track one. Initially the car went off the track where there was a sharp curve and low-contrast roadside. To make the car handle turns better, I added left and right camera images with corrective steering angles. Initially I compensated with a way too high steering angle, which caused the care to zigzag on the road instead of driving smoothly. Going from +-5.0 to +-0.3 compensation took care of that.

At the end of the process, the vehicle is able to drive autonomously around track 1 (the only track I trained it on) without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py, create_model()) consisted of a convolution neural network with the following layers and layer sizes:

- A normalization layer (160x320x3)
- A cropping layer to remove the sky and hood of the car (crop top 70 and bottom 25 pixels), out=65x320x3
- 5 convolutional layers, first half having half the output resolution of the previous layer but increasingly larger filter-sets
  - 24x31x158
  - 36x14x77
  - 48x5x37
  - 64x3x35
  - 64x1x31
- A flattening layer (2112 outputs)
- 4 dense/fully connected layers of increasingly smaller size perform classification, the last single-node layer outputs the steering angle
  - 100
  - 50
  - 10
  - 1 (the steering angle)

Throughout, the "elu" activation function is used. It is supposed to allow slightly faster training than the "relu" activation function. Use of an activation function introduces non-linearity, which is essential for learning non-linear/more abstract features.

Here is a visualization fo the architecture, copied from [NVIDIA's article](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025) about it:

![NVIDIA cnn architecture][model_visualization]

The layer sizes above applies to NVIDIA's original camera image sizes, Udacity's virtual car has somewhat higher resolution cameras.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, recorded several laps on track one using center lane driving, driving both directions to avoid bias of turning left or right. Here is an example image of center lane driving:

![center camera near start of track 1][center]

I also added the left and right camera images with +0.3 or -0.3 steering angle correction:

![corresponding left camera][left]
![corresponding right camera][right]

At this point the model performed well enough to safely drive several rounds of track 1 autonomously, and even though never trained for it, cleared much of the more difficult and different track 2 without prior practise.

If there was a need for future improvement, I would probably start with adding mirrored versions of the images. Then I would probably augment the images by adding tree-like shadows on random locations and of course, gathering more training data by driving more on both tracks.

After the collection process, I had 11097 times 3 (left/center/right) data points. There was no preprocessing done of the images besides steering-angle conversions. Normalization and clipping were done in the model itself.

I finally randomly shuffled the data set and put 20% of the data into a validation set. For each batch of 256 driving-log entries, I shuffled the data again to avoid having the same left/center/right camera order.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 5 as evidenced by no or little improvement of loss after that point. I used an adam optimizer so that manually setting or adjusting the learning rate wasn't necessary.
