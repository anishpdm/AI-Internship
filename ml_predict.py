import joblib
import numpy as np
pred_value = np.array( [[160]] ) 
result_model = joblib.load("my_model.pkl")
result= result_model.predict(pred_value)
print(result)