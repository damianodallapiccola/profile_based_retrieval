
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



def train_RF(X_train, y_train):
    print("-------------------")
    print("-- RANDOM FOREST --")
    print("-------------------")
    model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)
    model.fit(X_train, y_train)
    return model

def train_SVC(X_train, y_train):
    print("-------------------")
    print("------- SVM -------")
    print("-------------------")
    model = SVC(kernel='linear', gamma='auto')
    model.fit(X_train, y_train)
    return model

def eval_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    result = classification_report(y_val, y_pred)
    accSVC = accuracy_score(y_val, y_pred)
    print(result)
    print("\nAccuracy: ", accSVC)


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc






# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     NuSVC(probability=True),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier(),
#     GaussianNB(),
#     LinearDiscriminantAnalysis(),
 #     QuadraticDiscriminantAnalysis()]
