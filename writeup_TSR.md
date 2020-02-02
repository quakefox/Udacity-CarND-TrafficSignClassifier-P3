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


[//]: # (Image References)

[image1]: ./sample_images/bar_chart.png "barchart"
[image2]: ./sample_images/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./sample_images/img1.png "Traffic Sign 1"
[image5]: ./sample_images/img2.png "Traffic Sign 2"
[image6]: ./sample_images/img3.png "Traffic Sign 3"
[image7]: ./sample_images/img4.png "Traffic Sign 4"
[image8]: ./sample_images/img5.png "Traffic Sign 5"
[image9]: ./sample_images/conv1.png "Convolution Layer 1"
[image10]: ./sample_images/conv2.png "Convolution Layer 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 4

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is bar chart showing the distribution of sample images according to each labels in the training, validation and testing sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-processing

As a first step, I decided to convert the images to grayscale because the color information is not must have for traffic sign detection. Actually I have tested training CNN on both normalized RGB images and normalized grayscale images. The CNN trains much faster on grayscale images and have very good results.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a second step, I normalized the image by using (x - 128)/128 so that the grayscale pixel values are between -1 and 1. This way the dataset is balanced. 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Flatten               | 800                                           |
| Fully connected		| input: 800, output: 360        				|
| Dropout		        | Keep_prob = 0.5        				        |
| Fully connected		| input: 360, output: 168               		|
| Logits/Output			| input: 168, output: 43         				|
| Softmax				|												|
|						|												|
 



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, with the following hyper parameters:
batch size 128
epoch 30
learn rate 0.0015

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.1%
* test set accuracy of 93.3%

The LeNet architecture because it is used for image classifications. Minor changes were applied to this model to adjust the number of input and labels. More depth were given to the two convolution layers and two fully connected layers to provide more information since there are 43 number of classes, which is more than the 10 labels in the original LeNet architecture. Increased epoch from 10 to 30, this provides more data to train. From testing, it seems batch size of 128 yields better results than batch size of 256. The learn rate of 0.0015 is quite effective, gives faster convergence than 0.001 and avoids the large variance from 0.002.


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All five images were correctly detected!

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double Curve	      	| Double Curve				 				    |
| No Passing		    | No Passing									|
| Bicyle Crossing		| Bicyle Crossing      							|
| 60 km/h     			| 60 km/h 										|
| Go Straight or Left   | Go Straight or Left   						|

The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

For the 1st image, 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42         			| 21 - Double curve   							| 
| .37     				| 31 - Wild animals crossing 					|
| .13					| 27 - Pedestrians		                        |
| .12	      			| 11 - Right-of-way at the next intersection	|
| .10				    | 23 - Slippery road      						|

For the 2nd image,
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 2.33         			|  9 - No passing   							| 
| 1.20     				| 10 - No passing for vehicles over 3.5 tons 	|
| .56					| 20 - Dangerous curve to the right				|
| .49	      			| 41 - End of no passing					 	|
| .45				    | 23 - Slippery road |

For the 3rd image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| 29 - Bicycles crossing   						| 
| .32     				| 23 - Slippery road 							|
| .16					| 28 - Children crossing						|
| .08	      			| 20 - Dangerous curve to the right				|
| .07				    | 24 - Road narrows on the right      			|

For the 4th image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .50         			|  3 - Speed limit (60km/h)   					| 
| .20     				|  5 - Speed limit (80km/h) 		            |
| .19					| 38 - Keep right				                |
| .17	      			|  2 - Speed limit (50km/h)					 	|
| .05				    | 23 - Slippery road      				        |

For the 5th image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .65         			| 37 - Go straight or left   					| 
| .11     				| 40 - Roundabout mandatory 					|
| .09					| 26 - Traffic signals						    |
| .06	      			| 10 - No passing for vehicles over 3.5 tons	|
| .03				    | 23 - Slippery road      						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)


Visualization of the first convolution layer
![alt text][image9]

Visualization of the second convolution layer
![alt text][image10]

From the first two layers, it seems the CNN is capturing the edges and lines from the traffic signs on the first convolution layer, the information becomes denser at the second convolution layer.

I have difficulty visualizing the fully connected layers. Hope to figure out how to do that later.

