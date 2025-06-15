import pandas as pd
import matplotlib.pyplot as mt

df = pd.read_csv("data.csv")

X = df[ ["height"] ]
Y = df[ ["weight"] ] 

mt.scatter(X,Y)
mt.xlabel("Height in CM")
mt.ylabel("Weight in Kg")
mt.show()


