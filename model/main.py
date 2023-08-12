import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import pickle5 as pickle


def create_model(data):
    

    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy Score: ")
    print(accuracy_score(y_test, y_pred))

    print("Classification Report: \n", classification_report(y_test, y_pred))



    return model, scaler





def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(["id", "Unnamed: 32"], axis=1)

    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})


    return data

def main():
    data = get_clean_data()    
    
    model, scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler
    

if __name__ == "__main__":
        main()