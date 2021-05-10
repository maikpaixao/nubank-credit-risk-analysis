from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

class Model:
  def __init__(self, encoded_df):
    self.x = encoded_df.drop('target_default', axis=1).select_dtypes(exclude='object')
    self.y = encoded_df['target_default']
  
  def cross_validation(self, k=3):
    # Verificando nossa métrica com XGBClassifier
    # Com 5 repetições para cada divisão dos dados em 3 splits
    resultados = []
    for rep in range(5):
      print ('Repetição :',rep)
      kf = KFold(n_splits=k, shuffle=True, random_state = rep)
      
      for linhas_treino, linhas_teste in kf.split(self.x):
        print('Treino :', linhas_treino.shape[0])
        print('Teste :', linhas_teste.shape[0])

        X_train, X_test = self.x.iloc[linhas_treino], self.x.iloc[linhas_teste]
        y_train, y_test = self.y.iloc[linhas_treino], self.y.iloc[linhas_teste]

        ml_model = XGBClassifier(learning_rate=0.0145, n_estimators=1000, max_depth=6, subsample=1.0, colsample_bytree=1.0, gamma=1, 
                            random_state=0, n_jobs=1) # 84.84
        ml_model.fit(X_train, y_train)
        p = ml_model.predict(X_test)

        acc = np.mean(y_test == p)
        resultados.append(acc)
    return resultados