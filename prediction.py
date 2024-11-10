# code inspired by:
# https://365datascience.com/blog/authors/santiago-viquez/

import joblib # to load and save Python objects (incl. ML models)

def predict(data):
    clf = joblib.load("rf_model.sav") # loads the previously trained and saved ML model
    return clf.predict(data)
