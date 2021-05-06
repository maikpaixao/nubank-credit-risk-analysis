import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#reading data
data = pd.read_csv('data/training_data.csv')
#data['target_default'].apply(str)
data = data[['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'last_amount_borrowed', 'target_default']]

data['score_1'] = le.fit_transform(data['score_1'])
data['score_1'].astype('float64')

data['score_2'] = le.fit_transform(data['score_2'])
data['score_2'].astype('float64')

data['target_default'] = le.fit_transform(data['target_default'])
data['target_default'].astype('float64')

corr = data.corr()

#preprocessing
data = data.fillna(data.median())
print(data.info())

sns.heatmap(corr)
plt.show()
