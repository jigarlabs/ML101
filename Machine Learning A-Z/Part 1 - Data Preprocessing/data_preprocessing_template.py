# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt # Import sub library pyplot 
import pandas as pd

#Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.2,random_state=0 )

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # sc stands for scale. X vector
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)






 










