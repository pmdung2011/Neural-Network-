#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################

# headers =  ['age', 'sex','chest_pain','resting_blood_pressure',
#         'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
#         'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
#         'num_of_major_vessels','thal', 'heart_disease']

# 1 means "have heart disease" and 0 means "do not have heart disease"
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, dataFile, train_test_split_size, header=True, h=4):
        # np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        raw_data = pd.read_csv(dataFile)

        # TODO: Remember to implement the preprocess method

        # processed_data = self.preprocess(raw_data) # after processed: numpy type
        ncols = len(raw_data.columns)
        nrows = len(raw_data.index)
        # nrows, ncols = processed_data.shape
        self.X = raw_data.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        self.y = raw_data.iloc[:, (ncols - 1)].values.reshape(nrows, 1)
        # self.X = self.train_dataset[:, 0:(ncols - 1)]
        # self.y = self.train_dataset[:, (ncols - 1)]
        #

        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=train_test_split_size)
        self.X_train = self.preprocess(self.X_train)
        self.X_test = self.preprocess(self.X_test)

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #
    def __activation(self, x, activation):
        if activation == "sigmoid":
            return self.__sigmoid(x)
        elif activation == "tanh":
            return self.__tanh(x)
        elif activation == "relu":
            return self.__relu(x)
        return None

    #
    # TODO: Define the derivative function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(x)
        elif activation == "tanh":
            self.__tanh_derivative(x)
        elif activation == "relu":
            self.__relu_derivative(x)

    # SIGMOID
    '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
    '''

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):  # derivative of sigmoid function, indicates confidence about existing weight
        return x * (1 - x)

    # TANH(X)
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # RELU
    '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
    '''

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_derivative(self, x):  # derivative of Relu function (Assuming a derivative value of 0 for x=0)
        return (x > 0) * 1

    # Pre-process the dataset
    def preprocess(self, X):
        df = X
        # df = df.values
        sc = StandardScaler()  # standardize the dataset
        sc.fit(df)
        return sc.transform(df)

    # Below is the training function

    def train(self, activation, max_iterations, learning_rate):
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X_train, activation)
            error = 0.5 * np.power((out - self.y_train), 2)

            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, activation)

            update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
            update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X_train, 0), 1)).T, self.deltaOut)

            update_weight_hidden = learning_rate * np.dot(self.X_train.T, self.deltaHidden)
            update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X_train, 0), 1)).T, self.deltaHidden)

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))

    def forward_pass(self, input_data, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(input_data, self.W_hidden) + self.Wb_hidden
        # TODO: I have coded the sigmoid activation, you have to do the rest
        if activation == "sigmoid":
            self.X_hidden = self.__activation(in_hidden, activation)
            in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
            out = self.__sigmoid(in_output)

        elif activation == "tanh":
            self.X_hidden = self.__activation(in_hidden, activation)
            in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
            out = self.__tanh(in_output)

        elif activation == "relu":
            self.X_hidden = self.__activation(in_hidden, activation)
            in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
            out = self.__relu(in_output)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y_train - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y_train - out) * (self.__tanh_derivative(out))
        if activation == "relu":
            delta_output = (self.y_train - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation):
        prod = self.deltaOut.dot(self.W_output.T)
        if activation == "sigmoid":
            delta_hidden_layer = prod * (self.__sigmoid_derivative(self.X_hidden))
        elif activation == "tanh":
            delta_hidden_layer = prod * (self.__tanh_derivative(self.X_hidden))
        elif activation == "relu":
            delta_hidden_layer = prod * (self.__relu_derivative(self.X_hidden))

        self.deltaHidden = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation, header=True):
        out = self.forward_pass(self.X_test, activation)
        error = 0.5 * np.power((out - self.y_test), 2)
        return np.sum(error)

    # def plot_loss(self):
    #     '''
    #     Plots the loss curve
    #     '''
    #     plt.plot()
    #     plt.xlabel("Iteration")
    #     plt.ylabel("logloss")
    #     plt.title("Loss curve for training")
    #     plt.show()


if __name__ == "__main__":
    # perform pre-processing of both training and test part of the test_dataset
    # split into train and test parts if needed
    inp = input(
        "Choose Activation functions: \n 1 - Sigmoid \n 2 - Tanh \n 3 - Relu \n Sigmoid is Default Option \n Input --> ")
    if inp == "1":
        activation = "sigmoid"
    elif inp == "2":
        activation = "tanh"
    elif inp == "3":
        activation = "relu"
    else:
        activation = "sigmoid"

    # Specify the train_test split size (Recommend 0.10, 0.20 or 0.30)
    train_test_split_size = 0.001

    # Specify the number of iterations
    max_iterations = 5000

    # Specify the learning rate
    learning_rate = 0.001

    # Specify the dataset
    # https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat
    dataset_file = 'https://datasetdanny.s3.amazonaws.com/heart_data_uci.csv'

    # Invoke Neural Network class
    neural_network = NeuralNet(dataset_file, train_test_split_size)

    # Train the Neural Network
    neural_network.train(activation, max_iterations, learning_rate)

    # Test the Neural Network
    testError = neural_network.predict(activation)
    print("Iteration: ", max_iterations)
    print("Learning rate: ", learning_rate)
    print("Using Activation function: " + str(activation) + ", with error: " + str(testError))
