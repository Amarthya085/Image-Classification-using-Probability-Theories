from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras

# Function to create the CNN model
def create_model(filters, kernel_size, fc1_neurons, fc2_neurons, learning_rate, input_shape, num_classes=10):
    model = Sequential([
        Conv2D(filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        Conv2D(2 * filters, kernel_size=kernel_size, activation='relu', strides=(1, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        Flatten(),
        Dense(fc1_neurons, activation='relu'),
        Dense(fc2_neurons, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

# Function to train and evaluate the model
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=5):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)
num_classes = 10

x_train = x_train.reshape(x_train.shape[0], *input_shape).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], *input_shape).astype('float32') / 255

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Experiment 1: Explore the impact of kernel size on accuracy
model1 = create_model(filters=32, kernel_size=(3, 3), fc1_neurons=128, fc2_neurons=64, learning_rate=0.001,
                      input_shape=input_shape)
print("\nExperiment 1:")
print("Exploring the impact of kernel size on accuracy.")
train_and_evaluate_model(model1, x_train, y_train, x_test, y_test)

# Experiment 2: Explore the impact of the number of feature maps on accuracy
model2 = create_model(filters=64, kernel_size=(3, 3), fc1_neurons=128, fc2_neurons=64, learning_rate=0.001,
                      input_shape=input_shape)
print("\nExperiment 2:")
print("Exploring the impact of the number of feature maps on accuracy.")
train_and_evaluate_model(model2, x_train, y_train, x_test, y_test)

# Experiment 3: Explore the impact of the number of neurons in the fully connected layers on accuracy
model3 = create_model(filters=32, kernel_size=(3, 3), fc1_neurons=64, fc2_neurons=32, learning_rate=0.001,
                      input_shape=input_shape)
print("\nExperiment 3:")
print("Exploring the impact of the number of neurons in the fully connected layers on accuracy.")
train_and_evaluate_model(model3, x_train, y_train, x_test, y_test)

# Experiment 4: Explore the impact of learning rate on accuracy
model4 = create_model(filters=32, kernel_size=(3, 3), fc1_neurons=128, fc2_neurons=64, learning_rate=0.0001,
                      input_shape=input_shape)
print("\nExperiment 4:")
print("Exploring the impact of learning rate on accuracy.")
train_and_evaluate_model(model4, x_train, y_train, x_test, y_test)

# Experiment 5: Explore a combination of changes in parameters
model5 = create_model(filters=64, kernel_size=(5, 5), fc1_neurons=64, fc2_neurons=32, learning_rate=0.0001,
                      input_shape=input_shape)
print("\nExperiment 5:")
print("Exploring a combination of changes in parameters.")
train_and_evaluate_model(model5, x_train, y_train, x_test, y_test)
