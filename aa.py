import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDOneClassSVM

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

def get_dataset(name_train, name_test):
    train_dataset = pd.read_csv(name_train)
    test_dataset = pd.read_csv(name_test)
    train_dataset.head()
    test_dataset.head()
    return train_dataset, test_dataset

features = ['DailyRate','Education','DistanceFromHome']
target = ['Attrition']
X = train_dataset[features]
y = train_dataset[target]
print(X)
print(y)


X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2)

print(X_test)

# model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# model.fit(X_train, y_train)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# Dự đoán trên tập kiểm thử
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


test_private = pd.read_csv("public_test.csv")
label_test = pd.read_csv("public_test_with_labels.csv")


features = ['DailyRate','Education','DistanceFromHome']
target = ['Attrition']


public_test_with_labels = train_dataset[target]
public_test = train_dataset[features]

private_pred = model.predict(public_test)

print(private_pred)
print("Accuracy:", accuracy_score(private_pred , public_test_with_labels))