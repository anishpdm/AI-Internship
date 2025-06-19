import joblib
import pandas as pd


df = pd.read_csv("new_data.csv")

X = df[ ["height"] ]

pred_value = pd.DataFrame( [[160]] , columns=X.columns ) 
result_model_lr = joblib.load("lr_model.pkl")
result_model_knn = joblib.load("knn_model.pkl")


result= result_model_lr.predict(pred_value)
print(result)


result_knn= result_model_knn.predict(pred_value)
print(result_knn)