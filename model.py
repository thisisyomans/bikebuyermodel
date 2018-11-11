#imports for file reading and/or writing
from pandas import read_csv
from numpy import set_printoptions

#imports for different classifiers (more to be tested most probably)
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier

#reading the csv (comma separated values) file (currently set to the experimental dataset)
filename = 'data/BBCTest.csv'
dataframe = read_csv(filename) 

#defining the scope of the dataset to be used
array = dataframe.values
X = array[:,0:11] 
Y = array[:,11]

#defining training and testing
test_size = .30
seed = 45
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#setting model and fitting with training 
model = SVC()
model.fit(X_train, Y_train) 

#acquiring the result from testing
result = model.score(X_test, Y_test)

#printing the result as a percent
print(result*100)
