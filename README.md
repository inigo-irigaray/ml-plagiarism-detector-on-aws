# Plagiarism Detector, Machine Learning Deployment on AWS: testing SVM, PyTorch NN, Gaussian Process and Naive Baye's Classifers

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker. In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

## Files

This project will be broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.
* This first notebook is **not** required in your final project submission.

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Select "good" features, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker**

* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

**`source_sklearn`**

* `train_gpc.py`: Hyperparameter tuning job for a Gaussian Process Classifier, tuning for Isotropic and Anisotropic kernels.
* `train_nbc.py`: Hyperparameter tuning job for Naive Baye's Classifier, tuning for variance smoothing.
* `train_svc.py`: Hyperpatameter tuning job for a Support Vector Machine Classifier, tuning for kernel type (linear or radial basis function kernel), C and gamma (auto or scale).

**`source_pytorch`**

* `model.py`: Neural Net classifier model, composed of three layers, followed by ReLU nonlinearities and Dropout, except for the ouput layer, which uses only a sigmoid activation function for classification purposes.
* `predict.py`: Prediction loop for the SageMaker endpoint.
* `train.py`: Training loop for the model to be used by SageMaker.

## Setup

The notebooks and auxiliary files provided in this repository are intended to be executed using Amazon's SageMaker platform. The following is a brief set of instructions on setting up a managed notebook instance using SageMaker, from which the notebooks can be completed and run.

### Log in to the AWS console and create a notebook instance

Log in to the AWS console and go to the SageMaker dashboard. Click on `Create notebook instance`. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or object with sagemaker in the name is available to the notebook.

### Use git to clone the repository into the notebook instance

Once the instance has been started and is accessible, click on `open` to get the Jupyter notebook main page. We will begin by cloning this github repository into the notebook instance. Note that we want to make sure to clone this into the appropriate directory so that the data will be preserved between sessions.

Click on the `new` dropdown menu and select `terminal`. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repository as follows.

    cd SageMaker
    git clone https://github.com/inigo-irigaray/ml-plagiarism-detector-on-aws.git
    exit
    
After you've finished close the terminal window.
