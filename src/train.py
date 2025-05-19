
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from preprocess import load_data, preprocess_data

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(clf, "model.joblib")
    return acc

if __name__ == "__main__":
    acc = train_model()
    print(f"Model Accuracy: {acc:.2f}")

