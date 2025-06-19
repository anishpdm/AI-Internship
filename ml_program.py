import pandas as pd
import matplotlib.pyplot as mt
from sklearn import linear_model as lt
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor as kn


Y = df[ ["weight"] ] 

mt.scatter(X,Y)
mt.xlabel("Height in CM")
mt.ylabel("Weight in Kg")
mt.show()

X_train,X_test,Y_train,Y_test = train_test_split( X,Y, test_size=0.2 ,random_state=42 )



lr_model = lt.LinearRegression()
lr_model.fit(X_train,Y_train) #Training


knn_model = kn(n_neighbors=3)
knn_model.fit(X_train,Y_train)

#Training Over ---- 

joblib.dump(lr_model,"lr_model.pkl" ) #Saving To Model file

joblib.dump(knn_model,"knn_model.pkl" ) #Saving To Model file



print(" Training Completed ... !!!")


# Evaluation

print(X_test)

prediction_result_weight = lr_model.predict(X_test)

print(prediction_result_weight)

print( "LR MSE", mean_squared_error(Y_test,prediction_result_weight) )

print( "LR RMSE" , np.sqrt(mean_squared_error(Y_test,prediction_result_weight)) )

print ("LR r2 square" , r2_score(Y_test,prediction_result_weight) )

#Evaluation for KNN



prediction_result_weight_knn = knn_model.predict(X_test)
print( "KNN MSE", mean_squared_error(Y_test,prediction_result_weight_knn) )
print( "KNN RMSE" , np.sqrt(mean_squared_error(Y_test,prediction_result_weight_knn)) )
print ("KNN r2 square" , r2_score(Y_test,prediction_result_weight_knn) )















