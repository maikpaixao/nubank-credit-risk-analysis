import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

class Utils:
  def __init__(self, data):
    self.data = data
  
  def train(self):
    x = self.data.iloc[:, 1:]
    y = self.data['target_default']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

    model = xgb.XGBClassifier(n_jobs = 2, n_estimators = 200).fit(x_train, y_train)
    predict = model.predict(x_test)

    #return classification_report(predict, y_test)
    return accuracy_score(predict, y_test), precision_score(predict, y_test), recall_score(predict, y_test)#, f1_score(predict, y_test)