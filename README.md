# Convolutional Neural Networks
## Melanoma Detection Assignment

#### Problem Statement:

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


You can download the dataset here


The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

Actinic keratosis
Basal cell carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented benign keratosis
Seborrheic keratosis
Squamous cell carcinoma
Vascular lesion
 

NOTE:
You don't have to use any pre-trained model using Transfer learning. All the model building processes should be based on a custom model.
Some of the elements introduced in the assignment are new, but proper steps have been taken to ensure smooth learning. You must learn from the base code provided and implement the same for your problem statement.
The model training may take time to train as you will be working with large epochs. It is advised to use GPU runtime in Google Colab.
 

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
#### Business Goal:

Project Pipeline
Data Reading/Data Understanding → Defining the path for train and test images 
Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
Model Building & training : 
Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~20 epochs
Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
Model Building & training on the augmented data :
Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~20 epochs
Write your findings after the model fit, see if the earlier issue is resolved or not?
Class distribution: Examine the current class distribution in the training dataset 
- Which class has the least number of samples?
- Which classes dominate the data in terms of the proportionate number of samples?
Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
Model Building & training on the rectified class imbalance data :
Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~30 epochs
Write your findings after the model fit, see if the issues are resolved or not?

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
This is a neural network model that is being trained on some data. The model have 50 epochs and it has already gone through 21 epochs. For each epoch, there is a training set and a validation set. The model outputs the loss and accuracy for both the training set and the validation set.

In the first epoch, the model had a training loss of 1.9323 and training accuracy of 0.2650. The validation loss was 1.6189 and the validation accuracy was 0.4269. For the second epoch, the training loss was 1.5194 and the training accuracy was 0.4253. The validation loss was 1.3347 and the validation accuracy was 0.4744.

In general, the training loss and training accuracy decrease while the validation loss and validation accuracy increase as the model goes through more epochs. This suggests that the model is improving and becoming better at predicting the data. However, it's important to note that after a certain point, the model may start overfitting to the training data and performing worse on the validation data. Therefore, it's important to monitor the validation loss and accuracy closely to determine when the model starts overfitting.


<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.


## Contact
Created by [@githubusername] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->