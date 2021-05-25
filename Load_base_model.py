

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    #512 FC layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # Output layer
    tf.keras.layers.Dense(15, activation='softmax')
])


model.summary()

pre_trained_weights = r'Trained_models\BasicModel_250epochs.h5'

model.load_weights(pre_trained_weights)

#RUNNING THE MODEL
import numpy as np
from keras.preprocessing import image

def one_hot(index):
    array = np.zeros((1,15))
    array[0][index] = 1
    return array

tree_classes = [1,10,11,12,13,14,15,2,3,4,5,6,7,8,9]        #<-- Define the alphanumeric order of the classes, which is how tensorflow outputs the predictions
again = 'yes'
# os.chdir(r'C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\8vo Semestre\Intelligent Systems Technology\Final Project\Data\RealImages') #<-- Change directory to the one inside the parenthesis
os.chdir(r'C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\8vo Semestre\Intelligent Systems Technology\Final Project\Data\Test_set') #<-- Change directory to the one inside the parenthesis
Test_set_dir = r'C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\8vo Semestre\Intelligent Systems Technology\Final Project\Data\Test_set'

Incorrect_predictions = []
while (again == 'yes'):
    Correct_predictions = 0
    Accuracy = 0
    os.chdir(r'C:\Users\javie\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Universidad\8vo Semestre\Intelligent Systems Technology\Final Project\Data\Test_set') #<-- Change directory to the one inside the parenthesis
    number_of_folders = len(os.listdir())
    i = 0                                                                       #set counter which will tell me in which folder I am
    for Folder in os.listdir():
        current_path = os.path.join(Test_set_dir, Folder)
        os.chdir(current_path)
        Total_predictions = len(os.listdir()) * number_of_folders
        True_class = one_hot(i)                                                 #Create array of 14 0's and one 1 where the true class is in the i'th position

        for file in os.listdir():                                               # For each file in the current directory (Predict images)
            path = os.path.abspath(file)
            img = image.load_img(path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x/255
            images = np.vstack([x])
            classes = model.predict(images, batch_size=1)
            print(file)

            # print(True_class)
            Prediction_vector = one_hot(np.argmax(classes[0]))
            # print(Prediction_vector)

            index = np.argmax(classes[0])
            probability = classes[0][index] * 100
            print('Class {}, probability: {}'.format(tree_classes[index], probability))
            if ((Prediction_vector==True_class).all()):
                Correct_predictions += 1
            else:
                print('INCORRECTLY CLASSIFIED!\n')
                Incorrect_predictions.append(file)                              #Gets incorrectly classified examples
        i += 1
    Accuracy = Correct_predictions/Total_predictions
    print(Accuracy)
    for i in range(len(Incorrect_predictions)):
        print(Incorrect_predictions[i])
    again = input("Do you want to make another prediction?: ")
