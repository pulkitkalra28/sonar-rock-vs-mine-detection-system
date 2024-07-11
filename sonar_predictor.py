import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

sonar_data = pd.read_csv('sonardata.csv', header=None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: {:.2f}%'.format(training_data_accuracy * 100))

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data: {:.2f}%'.format(testing_data_accuracy * 100))

input_data = (
    0.0124, 0.0433, 0.0604, 0.0449, 0.0597, 0.0355, 0.0531, 0.0343, 0.1052, 0.2120, 0.1640, 0.1901, 0.3026, 0.2019,
    0.0592, 0.2390, 0.3657, 0.3809, 0.5929, 0.6299, 0.5801, 0.4574, 0.4449, 0.3691, 0.6446, 0.8940, 0.8978, 0.4980,
    0.3333, 0.2350, 0.1553, 0.3666, 0.4340, 0.3082, 0.3024, 0.4109, 0.5501, 0.4129, 0.5499, 0.5018, 0.3132, 0.2802,
    0.2351, 0.2298, 0.1155, 0.0724, 0.0621, 0.0318, 0.0450, 0.0167, 0.0078, 0.0083, 0.0057, 0.0174, 0.0188, 0.0054,
    0.0114, 0.0196, 0.0147, 0.0062)
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print('The object is a rock!')
else:
    print('The object is a mine!')
