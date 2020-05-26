from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV



# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

if __name__ == '__main__':
    # SageMaker parameters, like the directories for training data and saving models
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    # Define Gaussian Process Classifier and hyperparameter tuner
    gpc = GPC()
    model = GridSearchCV(estimator=gpc,
                        n_jobs=3,
                        verbose=10,
                        param_grid={'kernel': [1.0 * RBF([1.0]), 1.0 * RBF([1.0, 1.0, 1.0])]})
    
    # Train model and select best performing set of hyperparameters by default
    model.fit(train_x, train_y)
    
    print('Best Parameters: ', model.best_params_)
    print('Best Estimator: ', model.best_estimator_)
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))