import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pd

train_images_5= scipy.io.loadmat(r"D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project\training_data_5.mat")
images_5 = train_images_5["train_data_5"]
train_images_6= scipy.io.loadmat(r"D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project\training_data_6.mat")
images_6 = train_images_6['train_data_6']
#print(len(images))
''' to make sure that the images are stored in numpy arrays
for key, value in train_images_5/6.items():
    if isinstance(value, np.ndarray):
        print(f'Variable name: {key}, Data type: {type(value)}, Dimensions: {value.shape}')
'''

'''marge both the training data sets as we need to find mean and SD for the entire dataset'''
images = np.vstack((images_5, images_6))

'''
Now the array (numpy array) 'images' which con  sists the images is reshaped into a 2-d array 
i.e., the images are vectorized to a 784-d dimensioanl vector, each image is flattened into a 1-d vector (28*28->784) 
Now the size of vectorized_images is 892*784  (2-D array) for testing_data_5
'''
num_images, height, width = images.shape
vectorized_images = images.reshape(num_images, -1)

'''mean and SD for all the features in the '''
mean = np.mean(vectorized_images, axis=0)
#print(len(mean))
std = np.std(vectorized_images, axis=0)

''' 
Normalizatoin (feature xi-mean)/std 
normalized_images->numpy.ndarray, shape->(11339, 784)
'''
e = 1e-8  # Small constant to avoid division by zero
normalized_images = (vectorized_images - mean) / (std + e)

'''PCA'''
covariance_matrix = np.cov(normalized_images, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) #Eigen analysis

''' Desecnding order of eigen values and vectors '''
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_1 = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

selected_eigenvectors = eigenvectors[:, :2]

'''2-d projections of the samples on the first and second principal components'''
reduced_train_images = np.dot(normalized_images, selected_eigenvectors)

'''preprocessing testing samples'''
testing_images_5 = scipy.io.loadmat(r"D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project\testing_data_5.mat")
test_images_5 = testing_images_5['test_data_5']
testing_images_6 = scipy.io.loadmat(r"D:\Education\MS\Courses\Fundamentals of stat.. CSE569\project\testing_data_6.mat")
test_images_6 = testing_images_6['test_data_6']

test_images = np.vstack((test_images_5, test_images_6))
n,h,w = test_images.shape
vectorized_test_images = test_images.reshape(n, -1)
normalized_test_images = (vectorized_test_images - mean) / (std + e)

reduced_test_images = np.dot(normalized_test_images, selected_eigenvectors)

#2-D projections of Training and Testing images
projections_train_5 = reduced_train_images[:5421]
projections_train_6 = reduced_train_images[5421:]
projections_test_5 = reduced_test_images[:892]
projections_test_6 = reduced_test_images[892:]

'''
This code is to plot the projections of the training samples and testing samples

#plt.scatter(projections_train_5[:, 0], projections_train_5[:, 1], label='Training Samples of 5', c='b', marker='o')
#plt.scatter(projections_train_6[:, 0], projections_train_6[:, 1], label='Training Samples of 6', c='black', marker='x')

plt.scatter(reduced_train_images[:, 0], reduced_train_images[:, 1], label='Training Samples', c='b', marker='o')
plt.scatter(reduced_test_images[:, 0], reduced_test_images[:, 1], label='Testing Samples', c='black', marker='x')

plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('2-D Projection of Training and Testing Samples')
plt.legend(loc='best')

plt.show()
'''

'''
This code is used to verify whether the data is normally distributed or not

temp_df_class5 = pd.DataFrame(projections_train_5)
temp_df_class5 = temp_df_class5.rename(columns={0: 'PC-1', 1: 'PC-2'})
temp_df_class5['Class'] = 'Class 5'

temp_df_class6 = pd.DataFrame(projections_train_6)
temp_df_class6 = temp_df_class6.rename(columns={0: 'PC-1', 1: 'PC-2'})
temp_df_class6['Class'] = 'Class 6'

combined_df = pd.concat([temp_df_class5, temp_df_class6])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(data=combined_df, x='PC-1', hue='Class', common_norm=False, label='Class 5 and 6')
plt.title('Distribution for PC-1')

plt.subplot(1, 2, 2)
sns.kdeplot(data=combined_df, x='PC-2', hue='Class', common_norm=False, label='Class 5 and 6')
plt.title('Distribution for PC-2')

plt.show()
'''

#Parameters for each class
mean_train_5 = np.mean(projections_train_5,axis=0)
cov_matrix_train_5 = np.cov(projections_train_5, rowvar=False)
mean_train_6 = np.mean(projections_train_6,axis=0)
cov_matrix_train_6 = np.cov(projections_train_6, rowvar=False)

'''classification using bayesian decision theory'''

correct_classified_train_data = 0
correct_classified_test_data = 0

for i in range(len(reduced_train_images)):
    data_point = reduced_train_images[i] 
    class_likelihood_5 = multivariate_normal.pdf(data_point, mean_train_5, cov_matrix_train_5)
    class_likelihood_6 = multivariate_normal.pdf(data_point, mean_train_6, cov_matrix_train_6)
    #likelihood_max = max(class_likelihood_5,class_likelihood_6)
    predicted_class = 5 if class_likelihood_5>class_likelihood_6 else 6
    if ((predicted_class==5 and i<5421) or (predicted_class==6 and i>=5421)):
        correct_classified_train_data += 1
    
for i in range(len(reduced_test_images)):
    data_point = reduced_test_images[i] 
    class_likelihood_5 = multivariate_normal.pdf(data_point, mean_train_5, cov_matrix_train_5)
    class_likelihood_6 = multivariate_normal.pdf(data_point, mean_train_6, cov_matrix_train_6)
    #likelihood_max = max(class_likelihood_5,class_likelihood_6)
    predicted_class = 5 if class_likelihood_5>class_likelihood_6 else 6
    if ((predicted_class==5 and i<892) or (predicted_class==6 and i>=892)):
        correct_classified_test_data += 1

accuracy_training_data = correct_classified_train_data / len(reduced_train_images) *100
accuracy_testing_data = correct_classified_test_data / len(reduced_test_images) *100

print(f"Training Set Accuracy: {accuracy_training_data}%")
print(f"Testing Set Accuracy: {accuracy_testing_data}%")