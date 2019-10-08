# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt # Import sub library pyplot 
import pandas as pd

#Import dataset
dataset = pd.read_csv('Data.csv')

#Create a matrix of features or rather matrix of independent variables
# first ':' means we take all the lines i.e. rows and second ':-1' means
# We take all the columns except last one of the dataset
X = dataset.iloc[:,:-1].values

# DEpendent variable matrix, just the last column
Y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='median')

# Make imputer object fit to data
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encode the categorical columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

#Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.2,random_state=0 )


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # sc stands for scale. X vector
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)






 










