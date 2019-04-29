import pandas as pd
from data_preparation import prepare_dataset
from models import eval_model, test_model, train_RF, train_SVC
from demo import run_demo

# nltk.download('wordnet')


def main():
    # DATA PREPARATION
    dataset = pd.read_csv('dataset.csv', encoding="ISO-8859-1")
    X_train, X_val, X_test, y_train, y_val, y_test, vect = prepare_dataset(dataset)

    # MODELS TRAINING
    print("\n--------------------------------------------------------")
    print("------------------- MODELS  TRAINING -------------------")
    print("--------------------------------------------------------\n")

    # Random Forest
    modelRF = train_RF(X_train, y_train)
    eval_model(modelRF, X_val, y_val)
    # SVM
    modelSVC = train_SVC(X_train, y_train)
    eval_model(modelSVC, X_val, y_val)

    # MODELS TESTING
    print("\n--------------------------------------------------------")
    print("-------------- MODELS  TESTING (accuracy) --------------")
    print("--------------------------------------------------------\n")
    print("RANDOM FOREST:     ", test_model(modelRF, X_test, y_test))
    print("SVC:               ",test_model(modelSVC, X_test, y_test))

    run_demo(vect, modelSVC)



if __name__ == '__main__':
    main()
