import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def transform():
  data['score_1'] = le.fit_transform(data['score_1'])
  data['score_1'] = data['score_1'].astype('float64')

  data['score_2'] = le.fit_transform(data['score_2'])
  data['score_2'] = data['score_2'].astype('float64')

  data['job_name'] = le.fit_transform(data['job_name'])
  data['job_name'] = data['job_name'].astype('float64')

  data['target_default'] = le.fit_transform(data['target_default'])
  data['target_default'] = data['target_default'].astype('float64')

le = preprocessing.LabelEncoder()

#reading data
data = pd.read_csv('data/training_data.csv')
#data['target_default'].apply(str)

data = data[['target_default', 'score_1', 'score_2', 'score_3', 'risk_rate', 'last_amount_borrowed',
            'last_borrowed_in_months', 'credit_limit', 'income',  'job_name', 'n_bankruptcies', 
            'n_accounts', 'n_issues', 'external_data_provider_credit_checks_last_2_year']]

transform()
#corr = data.corr()

#preprocessing
data = data.fillna(data.median())
#print(data.info())

#sns.heatmap(corr)
#plt.show()

x = data.iloc[:, 1:]
y = data['target_default']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

model = xgb.XGBClassifier(n_jobs = 2, n_estimators = 200).fit(x_train, y_train)
predict = model.predict(x_test)
print(accuracy_score(predict, y_test))
