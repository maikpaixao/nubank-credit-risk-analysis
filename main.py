import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import Preprocess
from model import Model

if __name__ == '__main__':
  data = pd.read_csv('data/training_data.csv')
  preprocess = Preprocess(data)
  cleaned_data = preprocess.clean_data()

  model = Model(cleaned_data)
  score = model.cross_validation()
  
  print(np.mean(score))
