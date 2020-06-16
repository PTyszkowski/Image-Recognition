Introductiion 

Problem: reconaizing cloaths icons from Fashion MNIST dataset. Its a data set with 60,000 train examples and 10,000 test examples sized 28x28 pixels. Each icon belongs to one class of ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

Methods

Method used for data clasification is MLP (https://en.wikipedia.org/wiki/Multilayer_perceptron), where rectifier(https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is used as an activation finction and cost function is optimzied with Adam algorithm(https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam). Size of hidden layer and learning rate is chosen during validation process. 

Results

My accuracy on test data with hidden layer size 256 and learning rate 0.0002 is 0.8612.
Benchmark accuaracy for MLP Classifier is between 0.84 and 0.875.

Usage 

File is made for JupyterNotebook. Data has to be saved in folder 'data'. 
To be able to load data you also have to download mnist_reader.py file.
Program is based on Tensorflow machine learning library. You can get it here https://www.tensorflow.org/tensorboard/get_started
Full model will be aslo saved after running the program. 
Other libraries used are matplotlib and numpy. 
