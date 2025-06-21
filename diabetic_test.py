import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")

X= df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y =  df[["Outcome"]]
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y, test_size=0.2, random_state=42)

knn_model = kn(n_neighbors=3)

knn_model.fit(X_Train,Y_Train) # Algorithm 

decision_tree_model= DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_Train,Y_Train)


Y_prediction = knn_model.predict(X_Test)
print ("Accuracy Score of KNN ", accuracy_score(Y_Test,Y_prediction)*100)


Y_prediction_des_tree = decision_tree_model.predict(X_Test)
print ("Accuracy Score of Des Tree", accuracy_score(Y_Test,Y_prediction_des_tree)*100)


patient_data = pd.DataFrame( [[0,55,67,19,0,26.6,0.351,31]])

result = knn_model.predict(patient_data )

decision_tree_model_result = decision_tree_model.predict(patient_data)


if result[0]==1:
    print("Diabetic Person using KNN ")
else:
    
    print("Not a Diabetic Person using KNN ")


if decision_tree_model_result[0]==1:
    print("Diabetic Person using decision_tree_model ")
else:
    
    print("Not a Diabetic Person using decision_tree_model ")
