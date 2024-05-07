# Sentiment-Analysis-for-Memes-Combined-Image-Text-Model

# Introduction
The code implements a sentiment analysis pipeline for memes using deep learning techniques. It preprocesses a dataset containing images, captions, and sentiment labels, then defines neural network architectures for both images and text. These networks are combined into a single model, which is trained and evaluated using custom data loaders and loss functions. The training process involves iterating over batches of data, computing predictions, and updating model parameters via backpropagation. Finally, the script visualizes the model architecture using torchviz.

## 1. Libraries:
+ `pandas` and `numpy` for data manipulation and numerical operations.
+ `spacy` for natural language processing tasks such as text preprocessing.
+ `torch` for building and training neural networks using `PyTorch`.
+ `torchvision` for image transformation and dataset handling.
+ Other libraries like re for regular expressions, SentenceTransformer for text embeddings.

## 2. Preprocessing:
The preprocessing step involves cleaning and preparing the dataset for training:
+ It loads the dataset containing image filenames, captions, and labels.
+ It maps labels to numerical values for easier processing.
+ It applies text preprocessing functions like removing punctuation and converting text to lowercase.
+ It handles any missing values in the dataset.
+ Finally, it selects the necessary columns for further processing.

## 3. Data Loader:
Custom data loaders are defined to load images and corresponding captions from the dataset:
+ The `MemeSentimentDataset` class is created, which inherits from PyTorch's Dataset class.
+ This class is responsible for loading images, captions, and labels from the dataset.
+ Images are loaded using `PIL.Image` and transformed using `torchvision.transforms`.
+ The `__getitem__` method is defined to return a tuple containing image tensors, transformed text tensors, and label tensors.

## 4. Neural Network:
The script defines neural network architectures for both images and text:
+ Separate neural network classes (`NN for images and NN_text for text`) are created using PyTorch's nn.Module class.
+ These networks consist of linear layers followed by activation functions like ReLU.
+ The forward method defines the forward pass of the network, specifying how input data flows through the layers.

## 5. Combined model:
A combined model is created that takes input from both the image and text neural networks:
+ The `Combined_model` class combines the image and text neural networks using a series of linear layers.
+ It concatenates the output of both networks and passes it through additional linear layers to produce the final output.
+ The final output consists of predictions for different sentiment aspects such as humor, sarcasm.

## 6. Training and Testing:
Functions for training and testing the model are defined:
+ The `train_loop` function iterates over batches of training data, computes predictions, calculates loss, and performs backpropagation to update model parameters.
+ The `test_loop` function evaluates the model on the test data, computes loss and accuracy metrics, and prints the results.
+ These functions utilize PyTorch's autograd mechanism for automatic differentiation and optimization.

## 7. Visualization:
The script uses `torchviz` to visualize the computational graph of the model:
+ It generates a graphical representation of the model architecture, showing how data flows through the network.
+ This visualization helps understand the model structure and identify any potential issues in the architecture.
