# Dog Breed Classification

This repo is designed to be as easy as possible for machine learning hobbyists or professionals to learn how transfer learning works using a hands on example without installing a single piece of software. The students will get to use keras (with TensorFlow backend) along with GPUs on Huawei's ModelArts platform to classify different dog breeds.

### You'll still need these things:

- A web browser preferably chrome
- A Huawei cloud account with sufficient balance
- And an internet connection

### Data
[Kaggle's Dog Breed Datase](https://www.kaggle.com/c/dog-breed-identification/data)

#### Data Description:
- train.zip - the training set, you are provided the breed for these dogs
- test.zip - the test set, you must predict the probability of each breed for each image
- sample_submission.csv - a sample submission file in the correct format
- labels.csv - the breeds for the images in the train set

### We will:
- Preprocess images data for computer vision tasks
- Transfer Learning: build new layers on top of the pre-trained Xception model using Keras and Tensorflow
- Evaluate our model on the test set
- Run the model on new dog images from the web
