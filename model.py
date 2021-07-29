from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

class Model:
  def __init__(self, encoded_df):
    self.x = encoded_df.drop('target_default', axis=1).select_dtypes(exclude='object')
    self.y = encoded_df['target_default']
  
  def train(self, x_train, y_train):
    model = XGBClassifier(learning_rate=0.0145, n_estimators=200, max_depth=6,
                            subsample=1.0, colsample_bytree=1.0, gamma=1, random_state=0, n_jobs=1)
    model.fit(x_train, y_train)
    return model
  
  def cross_validation(self, k=3):
    resultados = []
    for rep in range(5):
      kf = KFold(n_splits=k, shuffle=True, random_state = rep)
      
      for linhas_treino, linhas_teste in kf.split(self.x):
        x_train, x_test = self.x.iloc[linhas_treino], self.x.iloc[linhas_teste]
        y_train, y_test = self.y.iloc[linhas_treino], self.y.iloc[linhas_teste]

        model = self.train(x_train, y_train)
        pred = model.predict(x_test)
        acc = np.mean(y_test == pred)
        resultados.append(acc)
    return resultados
