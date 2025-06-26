import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np



df = pd.read_csv("persons.csv")

genderEncoder = LabelEncoder()
BodyTypeEncoder = LabelEncoder() 

df["Gender_enc"] = genderEncoder.fit_transform(df["Gender"]) #Female -> 0 and Male -> 1
df["BodyType_enc"] = BodyTypeEncoder.fit_transform(df["BodyType"]) #Athletic -> 0 and Average -> 1 and Heavy -> 2 and Slim -> 3

X= df[["Age","Gender_enc","BodyType_enc","Height"]]
Y= df[["Weight"]]

print(X)
print(Y)

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

model = Sequential()
model.add(Dense(10, activation="relu", input_shape=(4,)))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit(X_train,Y_train, epochs=100)




Y_pred = model.predict(X_test)
mse= mean_squared_error(Y_test,Y_pred)
rmse= np.sqrt(mse)
r2_score= r2_score(Y_test,Y_pred)
print("R2 Score ", r2_score)

new_Input =np.array([[29,1,1,173]])
predicted_op= model.predict(new_Input)
print(predicted_op)







