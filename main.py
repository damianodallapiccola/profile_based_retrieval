import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


import nltk
nltk.download('wordnet')

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv', encoding = "ISO-8859-1")


x = dataset['news'].tolist()
y = dataset['type'].tolist()

for index,value in enumerate(x):
    print ("processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])


vect = TfidfVectorizer(stop_words='english', min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

print ("no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print ("train size:", X_train.shape)
print ("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# c_mat = confusion_matrix(y_test,y_pred)
# kappa = cohen_kappa_score(y_test,y_pred)
# acc = accuracy_score(y_test,y_pred)
#
#
# print ("Confusion Matrix:\n", c_mat)
# print ("\nKappa: ",kappa)
# print ("\nAccuracy: ",acc)


import random

PREFERENCES = ["business", "entertainment", "politics", "sport", "tech"]

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

def predict_doc_type(doc, vect):
    doc_cleaned = clean_str(doc)
    corpus = []
    corpus.append(doc_cleaned)
    test_vect = vect.transform(corpus)
    return model.predict(test_vect)[0]


users = ["Maria", "Nacho", "Luca", "Adam", "Tom"]
users_with_preferences = assign_preferences(users)

print("--------------------------------------------------------")
print("Users preferences:")
for user in users_with_preferences:
    print(user," -> ",users_with_preferences[user])

# Sport test article
# test_corpus = "The Lakers said in a statement that Walton and the team had “mutually agreed to part ways," \
#               "” without elaborating. The news came three days after Magic Johnson, the team’s president of " \
#               "basketball operations, made the surprise announcement that he was resigning so that he could " \
#               "devote more time to his various business interests."


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

        result = predict_doc_type(test_corpus, vect)
        print("--------------------------------------------------------")
        print("This article talks about", result, "and it's addressed to:")
        for user in users_with_preferences:
            if result in users_with_preferences[user]:
                print(user)
        continue






