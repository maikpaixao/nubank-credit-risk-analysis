import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from utils import Utils

le = preprocessing.LabelEncoder()

def transform():
  data['score_1'] = le.fit_transform(data['score_1'])
  data['score_1'] = data['score_1'].astype('float64')

  data['score_2'] = le.fit_transform(data['score_2'])
  data['score_2'] = data['score_2'].astype('float64')

  data['job_name'] = le.fit_transform(data['job_name'])
  data['job_name'] = data['job_name'].astype('float64')

  data['target_default'] = le.fit_transform(data['target_default'])
  data['target_default'] = data['target_default'].astype('float64')

data = pd.read_csv('data/training_data.csv')
data = data[['target_default', 'score_1', 'score_2', 'score_3', 'risk_rate', 'last_amount_borrowed',
            'last_borrowed_in_months', 'credit_limit', 'income',  'job_name', 'n_bankruptcies', 
            'n_accounts', 'n_issues', 'external_data_provider_credit_checks_last_2_year']]

#preprocessing
transform()
data = data.fillna(data.mean())
#corr = data.corr()

utils = Utils(data)
#acc = utils.train()

print(utils.train())
#sns.heatmap(corr)
#plt.show()
