#1. importing and manipulating data

import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head(10))

X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']
 

#adding bias column

m = X.shape[0]
a = np.ones((m, 1))
X.insert(loc = 0, column = 'Ones', value = a)                      #add ones for theta_0 



#change categorical data to continuous data

X.loc[X['Gender'] == 'Male', 'Gender_Male'] = 1                 #1 if male
X.loc[X['Gender'] == 'Female', 'Gender_Male'] = 0               #0 if female

del X['Gender']                                                 #delete intial gender column



#feature scaling

age_std = X['Age'].std()
age_ave = X['Age'].mean()

sala_std = X['EstimatedSalary'].std()
sala_ave = X['EstimatedSalary'].mean()

X['Age'] = (X['Age'].subtract(age_ave)).divide(age_std)
X['EstimatedSalary'] = (X['EstimatedSalary'].subtract(sala_ave)).divide(sala_std)



#create train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)     #75% train, 25% test
X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()  #convert df to np for matrix operations




#2. functions of the algorithm

#sigmoid to convert numbers between 0 and 1
def sigmoid(z):
	return (1/(1+np.exp(-z)))



#hypothesis
def h(theta, X):
	return sigmoid(np.matmul(X, theta))
	


#cost function
def cost_function(X, y, theta, m):

	y = y.reshape(y.shape[0], 1)
	H = h(theta, X)	

	return (sum((y)*np.log(H) + (1-y)*np.log(1-H))) / (-m)

	

#grading descent
def gradient_descent(theta, X, y, alfa, m):

	H = h(theta, X)
	H = H.reshape((H.shape[0],))

	diff = np.subtract(H, y)
	a = np.matmul(np.transpose(X), diff).reshape((theta.shape[0],1))
	
	theta = theta - (alfa/m) * a

	return theta



#train function
def train(X, y, theta, alfa, m, num_iter):

	for i in range(num_iter):
		theta = gradient_descent(theta, X, y, alfa, m)

	
		if i % 500 == 0:
			print("Cost: ", cost_function(X, y, theta, m))

	return theta



#returns 1 or 0 with given threshold
def predict(X, theta, threshold = 0.5):
	
	a = h(theta, X)
	a [a >= threshold] = 1
	a [a < threshold]  = 0 

	return a



#returns the accuracy 
def score(y1, y2):

	#y1 is answer
	#y2 is calculated

	y1 = y1.reshape(y1.shape[0], 1)
	y2 = y2.reshape(y2.shape[0], 1)

	y1_not = (1 - y1).reshape(y1.shape[0], 1)
	y2_not = (1 - y2).reshape(y1.shape[0], 1)

	
	a = np.multiply(y1_not, y2_not) + np.multiply(y1, y2)   #1 means correct prediction, 0 means wrong predictions
	
	ones_ = np.count_nonzero(a == 1)  #count ones to get the percentage

	return (ones_ / y1.shape[0]) * 100




#3. initialization and function calls

m = X_train.shape[0]  #number of rows
n = X_train.shape[1]  #number of columns

theta = np.zeros((n, 1))
num_iter = 2000
alfa = 0.3

opt_theta = train(X_train, y_train, theta, alfa, m, num_iter)
y_ = predict(X_test, opt_theta)
print("Accuracy: ", score(y_test, y_))