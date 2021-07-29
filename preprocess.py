import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
sns.set_style()

class Preprocess:
  def __init__(self, data):
      self.data = data
  
  def encode_labels(self):
    encoded_df = self.data.copy()
    cat_cols = encoded_df.select_dtypes('object').columns
    
    for col in cat_cols:
      encoded_df[col+'_encoded'] = LabelEncoder().fit_transform(encoded_df[col])
      encoded_df.drop(col, axis=1, inplace=True)

    return encoded_df

  def normalize_data(self):
    scaled_df = self.data.copy()
    num_cols = scaled_df.drop('target_default',
                            axis=1).select_dtypes(exclude='object').columns

    scaled_df[num_cols] = StandardScaler().fit_transform(scaled_df[num_cols].values)
    return scaled_df

  def clean_data(self):
    df_clean = self.data.copy()

    df_clean.drop(labels=['ids', 'external_data_provider_credit_checks_last_year', 'channel', 'profile_phone_number'],
                    axis=1, inplace=True)

    df_clean['reported_income'] = df_clean['reported_income'].replace(np.inf, np.nan)
    df_clean.loc[df_clean['external_data_provider_email_seen_before'] == -999.0,
                'external_data_provider_email_seen_before'] = np.nan

    drop_var = ['reason', 'zip', 'job_name', 'external_data_provider_first_name',
                'lat_lon', 'shipping_zip_code', 'user_agent', 'profile_tags',
                'application_time_applied', 'email', 'marketing_channel',
                'shipping_state', 'target_fraud']

    df_clean.drop(labels=drop_var, axis=1, inplace=True)
    df_clean.dropna(subset=['target_default'], inplace=True)
    df_clean.last_amount_borrowed.fillna(value=0, inplace=True)
    df_clean.last_borrowed_in_months.fillna(value=0, inplace=True)
    df_clean.n_issues.fillna(value=0, inplace=True)

    num_df = df_clean.select_dtypes(exclude='object').columns
    cat_df = df_clean.select_dtypes(include='object').columns

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer = imputer.fit(df_clean.loc[:,cat_df])
    df_clean.loc[:,cat_df] = imputer.transform(df_clean.loc[:,cat_df])

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer = imputer.fit(df_clean.loc[:,num_df])
    df_clean.loc[:,num_df] = imputer.transform(df_clean.loc[:,num_df])

    return df_clean
    