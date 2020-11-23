import os.path
import numpy as np
import pandas as pd
import pickle
import joblib

from .utility import clean, stopwords

# Relative paths
BASE = os.path.dirname(os.path.abspath(__file__))

class NewsPredictor:

    def __init__(self, vectorizer_file, model_file):

        self.model          = joblib.load(os.path.join(BASE,model_file))
        self.vectorizer     = joblib.load(os.path.join(BASE,vectorizer_file))
        self.data           = None


    def load_and_clean_data(self, data):

        data = clean(data)
        self.data               = self.vectorizer.transform([data])


    def predict(self):

        if self.data is not None:
            prob                = self.model.predict_proba(self.data)
            prediction          = self.model.predict(self.data)

            return {'prob':prob[0].tolist(),'prediction':int(prediction)}
