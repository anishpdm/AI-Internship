import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("persons.csv")

genderEncoder = LabelEncoder()
BodyTypeEncoder = LabelEncoder() 

df["Gender_enc"] = genderEncoder.fit_transform(df["Gender"]) #Female -> 0 and Male -> 1
df["BodyType_enc"] = BodyTypeEncoder.fit_transform(df["BodyType"]) #Athletic -> 0 and Average -> 1 and Heavy -> 2 and Slim -> 3

X= df[["Age","Gender_enc","BodyType_enc","Height"]]
Y= df[["Weight"]]

print(X)