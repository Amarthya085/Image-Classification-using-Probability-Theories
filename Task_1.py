import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def read_data(data_file, label, normalize=True):
    # Read data from the file and parse it into samples
    with open(data_file, "rb") as f:
        lines = f.readlines()
    samples = [tuple(map(float, line.decode('UTF-8').strip('\n').split())) for line in lines]
    
    # Convert samples to NumPy array
    x = np.array(samples)

    # Normalize the data if specified
    x = (x - x.mean(axis=0))/x.std(axis=0) if normalize else x

    # Create labels based on the specified class (0 or 1)
    y = np.zeros((x.shape[0], 1), dtype='int') if label == 0 else np.ones((x.shape[0], 1), dtype='int')

    # Combine features and labels horizontally
    return np.hstack((x, y))

class DataHandler:
    _inst = None

    @staticmethod
    def get_inst():
        # Singleton pattern: create an instance if none exists, else return the existing one
        if DataHandler._inst is None:
            DataHandler._inst = DataHandler()
        return DataHandler._inst

    def __init__(self):
        # Ensure the class is initialized only once
        if hasattr(self, "_init"):
            return
        else:
            self._init = True

        # Initialize data attributes as None
        self.training_data, self.validation_data, self.testing_data = None, None, None

        # Load training and validation data
        self.load_train_and_val()

        # Load testing data
        self.load_test()

    def load_train_and_val(self, is_normalized=True):
        try:
            # Load and normalize training data for class 0
            train_p1 = read_data(data_file="D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project_part2\data_files\Train1.txt", label=0, normalize=is_normalized)
            # Load and normalize training data for class 1
            train_p2 = read_data(data_file="D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project_part2\data_files\Train2.txt", label=1, normalize=is_normalized)

            # Combine data from both classes to form training and validation sets
            self.training_data = np.vstack((train_p1[:1500, :], train_p2[:1500, :]))
            self.validation_data = np.vstack((train_p1[1500:, :], train_p2[1500:, :]))
        except Exception as e:
            print(f"Error loading training and validation data: {e}")
            # Handle the error as needed

    def load_test(self, is_normalized=True):
        try:
            # Load and normalize testing data for class 0
            test_p1 = read_data(data_file="D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project_part2\data_files\Test1.txt", label=0, normalize=is_normalized)
            # Load and normalize testing data for class 1
            test_p2 = read_data(data_file="D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project_part2\data_files\Test2.txt", label=1, normalize=is_normalized)

            # Combine data from both classes to form the testing set
            self.testing_data = np.vstack((test_p1, test_p2))
        except Exception as e:
            print(f"Error loading testing data: {e}")
            # Handle the error as needed

    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data

    def get_testing_data(self):
        return self.testing_data


class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        # Initialize the neural network with specified input, hidden, and output layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize data placeholders for training, validation, and testing sets
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None
        self.test_data = None
        self.test_labels = None

        # Initialize the neural network layers
        self.layers = []
        self.ini_network()
        self.Hidden_L_idx = 0
        self.Out_L_idx = 1

    def ini_network(self):
        # Initialize the weights and outputs for the hidden and output layers
        Hidden_L = [{'weights': np.random.rand(self.input_size + 1), 'output': 0} for _ in range(self.hidden_size)]
        Out_L = [{'weights': np.random.rand(self.hidden_size + 1), 'output': 0} for _ in range(self.output_size)]

        # Append the layers to the neural network
        self.layers.append(Hidden_L)
        self.layers.append(Out_L)

    def f_prop(self, sample):
        # Perform forward propagation through the neural network
        inputs = np.hstack(([1], sample[:]))

        for layer in self.layers:
            outputs = [max(0, np.dot(neuron['weights'], inputs)) for neuron in layer]
            for i, neuron in enumerate(layer):
                neuron['output'] = outputs[i]

            inputs = np.hstack(([1], outputs[:]))

        return inputs[1:]

    def predict(self, data):
        # Make predictions on the given dataset
        return np.array([np.round(self.f_prop(sample)[0]) for sample in data])

    def b_prop(self, true_label):
        # Perform backpropagation to update weights based on the error
        Out_L = self.layers[self.Out_L_idx]

        for neuron in Out_L:
            error = true_label - neuron['output']
            neuron['delta'] = error if neuron['output'] > 0 else 0

        Hidden_L = self.layers[self.Hidden_L_idx]

        for n in range(len(Hidden_L)):
            error = sum(o_neuron['weights'][n + 1] * o_neuron['delta'] for o_neuron in Out_L)
            Hidden_L[n]['delta'] = error if Hidden_L[n]['output'] > 0 else 0

    def get_error(self, data, labels):
        # Calculate the mean squared error for the given dataset
        total_error = sum(np.sum(np.square(self.f_prop(sample) - ground_truth)) for sample, ground_truth in zip(data, labels))
        return total_error / data.shape[0]

    def update_weights(self, sample, lr):
        # Update weights based on the calculated delta values
        inputs = np.hstack(([1], sample[:]))

        for layer in self.layers:
            outputs = [neuron['output'] for neuron in layer]
            for i, neuron in enumerate(layer):
                neuron['weights'] += neuron['delta'] * inputs * lr
            inputs = np.hstack(([1], outputs[:]))

    def get_validation_error(self):
        # Get the mean squared error on the validation set
        return self.get_error(self.val_data, self.val_labels)

    def get_testing_error(self):
        # Get the mean squared error on the testing set
        return self.get_error(self.test_data, self.test_labels)

    def train(self, learning_rate=9e-2, n_epochs=100, batch_size=-1):
        # Train the neural network using backpropagation
        epoch = 0
        training_error_history = []
        validation_error_history = []
        testing_error_history = []

        while epoch < n_epochs:
            epoch += 1
            training_error = 0.0
            for sample, ground_truth in zip(self.train_data, self.train_labels):
                output = self.f_prop(sample)
                training_error += np.sum(np.square(output - ground_truth))
                self.b_prop(ground_truth)
                self.update_weights(sample, learning_rate)

            training_error = training_error / self.train_data.shape[0]
            val_error = self.get_validation_error()
            testing_error = self.get_testing_error()
            
            print("[epoch: {}; learning_rate: {}; training_error: {}; validation_error: {}]"
                  .format(epoch, learning_rate, training_error, val_error))
            training_error_history.append(training_error)
            validation_error_history.append(val_error)
            testing_error_history.append(testing_error)

        # Plot the training, validation, and testing error history
        plt.title("Multi Layer Perceptron: {} x {} x {}".format(self.input_size, self.hidden_size, self.output_size))
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (J/n)")  # Change ylabel to represent Mean Squared Error
        plt.plot(training_error_history, label="Training", color="blue")  # Set the color to blue
        plt.plot(validation_error_history, label="Validation", color="green")  # Set the color to green
        plt.plot(testing_error_history, label="Testing", color="red")  # Set the color to red
        plt.legend()
        figure = plt.gcf()
        plt.show()



def train(neural_net):
    # Train the neural network and evaluate on the testing set
    train_data = DataHandler.get_inst().get_training_data()
    np.random.shuffle(train_data)
    x_train = train_data[:, :2]
    y_train = train_data[:, 2]
    neural_net.train_data = x_train
    neural_net.train_labels = y_train
    val_data = DataHandler.get_inst().get_validation_data()
    np.random.shuffle(val_data)
    x_val = val_data[:, :2]
    y_val = val_data[:, 2]
    neural_net.val_data = x_val
    neural_net.val_labels = y_val
    test_data = DataHandler.get_inst().get_testing_data()
    x_test = test_data[:, :2]
    y_test = test_data[:, 2]
    neural_net.test_data = x_test
    neural_net.test_labels = y_test
    neural_net.train(learning_rate=0.01, n_epochs=50)
    y_pred = neural_net.predict(x_test)
    res = accuracy_score(y_test, y_pred)
    print("Accuracy on testing data: {}".format(res))
    return res

hidden_sizes = [4,8,10,12,14]

for i in hidden_sizes:
    nn = NeuralNetwork(input_size=2, hidden_size=i, output_size=1)
    train(nn)