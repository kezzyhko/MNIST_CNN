# MNIST_CNN
Yet another convolutional neural network trained on MNIST

## About
This project was made for [the contest](https://competitions.codalab.org/competitions/26282), which was a part of Introduction to Machine Learning course in Innopolis University. The model was tested on highly augmented data. The final F1 score of the last model was 0.8924. The commit messages contain scores of each model. The provided code can produce model with different weights, and thus, different score.

## Files in this repository
* `IMLProject.ipynb` - the notebook that was used to train the model. It can also be found at [Google Colab](https://colab.research.google.com/drive/1FJ7nZiczW993RLw666i9NGO1T5_P2Vhh).
* `code.py` - standalone python file, should work the same as `.ipynb` notebook, but it was never actually tested.
* `model.h5.zip` - the zip file, containing only `model.h5` file - the model itself.
* Model is trained on augmented MNIST dataset. Unfortunately, dataset with augmentations was not saved during testing.
