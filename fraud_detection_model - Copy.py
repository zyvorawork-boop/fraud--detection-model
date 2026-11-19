# Fraud Detection Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# step 1 : Load CSV File
df = pd.read_csv("fraud_detection_data.csv")
print(df.head())
# step 2 : Feature and Target
X = df[["amount", "hour","is_international","new_device","failed_attempts"]]
y = df["is_fraud"]
# step 3 : Train Test Split
X_train, X_test,y_train, y_test = train_test_split(X,y , test_size = 0.2 , random_state = 42)
# step 4 : Create Model

model = RandomForestClassifier(random_state = 42)
# step 5 : Train Model

model.fit(X_train,y_train)
# step 6 : Prediction

y_pred = model.predict(X_test)
# step 7 : Accurancy

print("Accurancy :" , accuracy_score(y_test, y_pred))
# step 8 : Confusion Matrix

print("Confusion Matrix :")
print(confusion_matrix(y_test,y_pred))
# step 9 : New Transcation Prediction

amount = float(input("Enter Transcation Amount:"))
hour = int(input(" Enter your hours:"))
is_international = int(input("Is it from international:"))
new_device = int(input(" Is from a new device:"))
failed_attempt = int(input("How many failed attempt:"))
# put values 
new_data = pd.DataFrame({
            "amount" : [amount],
            "hour" : [hour],
            "is_international" : [is_international],
            "new_device" : [new_device],
            "failed_attempts" :[failed_attempt]})
result = model.predict(new_data)
if result[0] == 1:
    print("Fraud Transcation")
else:
    print("Normal Transcation")