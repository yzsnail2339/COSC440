# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)
No, code working well

Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)
To speed up gradient descent.
Avoiding excessively large or small numerical values.

Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)
It allows the model to handle higher dimensions and have different biases for different dimensions, rather than always passing through the origin.

Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)
Updating when using PyTorch or TensorFlow is easier to understand, making the code more readable and the logic clearer.

Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)
The data has been processed into grayscale values, the size of each image is uniform, the dataset is large enough, and the data has been correctly labeled.

Q6: Suppose you are an administrator of the NZ Health Service (CDHB or similar). What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize numerical codes on forms that are completed by hand by a patient when arriving for a health service appointment? (2-4 sentences)
positive：
Reducing the workload while adding human verification can ensure higher accuracy.
negative:
If employees overly trust machine recognition without careful verification, it can lead to errors due to not-so-high accuracy, posing certain risks.