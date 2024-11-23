### Image Model
## Class Definition
- Defines a custom dataset class for the FER2013 dataset, which is typically used for facial emotion recognition.
- Allows for loading and processing the FER2013 dataset efficiently for facial emotion recognition tasks. It extracts images and labels from a DataFrame, processes the image pixels, optionally applies transformations, and returns the image and its label.
## Loading Data, Splitting the Data
- Transforming data in preparation for training a machine learning model.
## Setting Data Transformations
- Resizing ensures that all images have the same dimensions, which is a requirement for feeding them into a neural network.
## Creating the Dataset Objects
- Defines how to access and preprocess the data.
- These dataset objects encapsulate that logic, allowing them to be passed to a DataLoader.
## Creating the DataLoader Objects
- Easy to load data in batches and shuffle it if necessary.
- Batches : Helps to compute gradients more efficiently and improves training stability.
- Shuffling the training data ensures that the model is not biased by the order in which it sees the data, leading to better generalization.
## Loading a Pre-trained ResNet Model
- Using a pre-trained model speeds up the training process, as the model already has general knowledge about image features.
## Modifying the Output Layer
- Seven emotion categories.
- Since the original ResNet model was trained for 1000 ImageNet classes, it needs to be adapted for the 7 emotion classes in the FER2013 dataset.
## Defining the Loss Function and Optimizer
- Adam is often used because it combines the advantages of both the AdaGrad and RMSProp optimizers. It adapts the learning rate for each parameter and is often faster and more efficient than other optimization algorithms.
## Training Phase and Evaluation Phase
- This code snippet defines a function train_model for training and evaluating a deep learning model using PyTorch. It includes both the training phase (where the model learns from the training data) and the evaluation phase where the model is tested on the validation data to check its performance
- Puts the model into "training mode", which is necessary for layers like dropout and batch normalization to behave correctly during training.
## Test Model
- This code snippet defines a function test_model for evaluating the trained model on the test dataset and calculating its accuracy. The test set is typically used for assessing how well the model generalizes to new, unseen data. 
- Check the model's performance on unseen data after training, providing a final assessment of how well it has learned to classify the images.