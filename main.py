import random
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from textblob import Word

#nltk.download('wordnet')


PREFERENCES = ["business", "entertainment", "politics", "sport", "tech"]



def clean_str(string):
    """
    String cleaning for datasets.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def prepare_dataset(dataset):
    # devide in features and labels
    x = dataset['news'].tolist()
    y = dataset['type'].tolist()
    print("Processing data...")
    for i, value in enumerate(x):
        x[i] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
    print("Done!")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    vect = TfidfVectorizer(stop_words='english', min_df=2)

    X_train = vect.fit_transform(X_train)
    y_train = np.array(y_train)
    X_test = vect.transform(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, vect


def assign_preferences(users):
    users_with_preferences = {}
    for user in users:
        preferences = []
        possibilities = list(PREFERENCES)
        for pref in range(random.randint(1, 3)):
            category = random.choice(possibilities)
            preferences.append(category)
            possibilities.remove(category)
        users_with_preferences[user] = preferences
    return users_with_preferences

def predict_doc_type(doc, vect, model):
    doc_cleaned = clean_str(doc)
    corpus = []
    corpus.append(doc_cleaned)
    test_vect = vect.transform(corpus)
    return model.predict(test_vect)[0]

# Testing results
def test(vect, model):
    users = ["Maria", "Nacho", "Luca", "Adam", "Tom", "Mike"]
    users_with_preferences = assign_preferences(users)

    print("--------------------------------------------------------")
    print("Users preferences:")
    for user in users_with_preferences:
        print(user," -> ",users_with_preferences[user])

    while True:
        try:
            print("--------------------------------------------------------")
            test_corpus = input("Paste here an article without newline characters (\\n) (Press 'q' to quit or 'r' to reassign the preferences): ")
        except ValueError:
            continue
        if test_corpus == "q":
            break
        if test_corpus == "r":
            users_with_preferences = assign_preferences(users)
            print("--------------------------------------------------------")
            print("Users preferences:")
            for user in users_with_preferences:
                print(user, " -> ", users_with_preferences[user])
            continue
        else:

            result = predict_doc_type(test_corpus, vect, model)
            print("--------------------------------------------------------")
            print("This article talks about", result.upper(), "and it's addressed to:")
            for user in users_with_preferences:
                if result in users_with_preferences[user]:
                    print(user)
            continue


# RUNTIME
dataset = pd.read_csv('dataset.csv', encoding = "ISO-8859-1")

X_train, X_test, y_train, y_test, vect = prepare_dataset(dataset)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

print("Train set:"+ str(X_train.shape))
print("Validation set:"+ str(X_val.shape))
print("Test set:"+ str(X_test.shape))


# Random Forest
print("RANDOM FOREST")
modelRF = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
modelRF.fit(X_train, y_train)
y_predRF = modelRF.predict(X_val)
accRF = classification_report(y_val,y_predRF)
print(accRF)


# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     NuSVC(probability=True),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier(),
#     GaussianNB(),
#     LinearDiscriminantAnalysis(),
#     QuadraticDiscriminantAnalysis()]


"""
# Random Forest
print("RANDOM FOREST")
modelRF = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
modelRF.fit(X_train, y_train)
y_predRF = modelRF.predict(X_test)
accRF = accuracy_score(y_test,y_predRF)
print("\nAccuracy: ",accRF)
"""



test(vect, modelRF)







