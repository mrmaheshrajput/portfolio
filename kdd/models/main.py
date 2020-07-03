import os.path
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .utility import red_feat_index, col_names

BASE = os.path.dirname(os.path.abspath(__file__))


class CustomScaler(BaseEstimator,TransformerMixin):

    def __init__(self,columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled.reset_index(drop=True), X_scaled], axis=1)[init_col_order]


class ScoreModel():

    def __init__(self, churn_model, appetency_model, upselling_model, scaler_file,freq_file):

        # with open(os.path.join(BASE,'churn'), 'rb') as churn_model, open(os.path.join(BASE,'appetency'), 'rb') as appetency_model, open(os.path.join(BASE,'upselling'), 'rb') as upselling_model, open(os.path.join(BASE,'scaler'), 'rb') as scaler_file, open(os.path.join(BASE,'freq_encodings'), 'rb') as freq_file:
        with open(os.path.join(BASE,'churn'), 'rb') as churn_model, open(os.path.join(BASE,'appetency'), 'rb') as appetency_model, open(os.path.join(BASE,'upselling'), 'rb') as upselling_model, open(os.path.join(BASE,'freq_encodings'), 'rb') as freq_file:
            self.churn = pickle.load(churn_model)
            self.appetency = pickle.load(appetency_model)
            self.upselling = pickle.load(upselling_model)
            # obj = CustomScaler(col_names)
            # self.scaler = pickle.load(scaler_file)
            self.freq_encodings = pickle.load(freq_file)
            self.scaler = joblib.load(os.path.join(BASE, scaler_file))
            self.data = None
            self.upselling_data = None

        self.scaler = joblib.load(os.path.join(BASE,'scaler'))

    def load_and_clean_data(self, data):
        missing_indicator = (data == None).astype('int64')
        dat = np.concatenate((np.delete(data,red_feat_index), missing_indicator),axis=0)
        dat = np.concatenate((dat,[len(np.where(dat == None)[0]), len(np.where(dat == 0)[0])]),axis=0)
        assert(len(dat) == 299)

        pro_data = np.array([np.where(type(i)==str, self.freq_encodings.get(i),i).ravel()[0] for i in dat]).astype('float')

        df=pd.DataFrame(pro_data.reshape(1,-1), columns=col_names)

        scaled_inputs = self.scaler.transform(df.iloc[:,:39])[0]
        for i,col in enumerate(df.iloc[:,:39].columns.values):
            df[col] = scaled_inputs[i]
#         ok_df = self.scaler.transform(df)
        assert(df.shape[1] == 299)

        df = df[self.churn.get_booster().feature_names]
        up_df = df[self.upselling.get_booster().feature_names]

        self.data = df
        self.upselling_data = up_df

    def predict(self):

        if self.data is not None:
            churn_prob = self.churn.predict_proba(self.data)[0]
            churn_pred = self.churn.predict(self.data)[0]

            appetency_prob = self.appetency.predict_proba(self.data)[0]
            appetency_pred = self.appetency.predict(self.data)[0]

            upselling_prob = self.upselling.predict_proba(self.upselling_data)[0]
            upselling_pred = self.upselling.predict(self.upselling_data)[0]

            predictions = {'churn': {'prob':churn_prob.tolist(),'pred':int(churn_pred)},
                          'appetency': {'prob':appetency_prob.tolist(),'pred':int(appetency_pred)},
                          'upselling': {'prob':upselling_prob.tolist(),'pred':int(upselling_pred)}
                          }

            return predictions
