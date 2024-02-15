## IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/Users/bijua/Downloads/Datasets/Algerian_forest_fires_dataset.csv")


# Region 1 = Bejaia Region
df.loc[:122,'Region']=1
#Region 2 = Sidi-Bel Abbes Region
df.loc[122:,'Region']=2
df[['Region']] = df[['Region']].astype(int)


# remove null value
df=df.dropna().reset_index(drop=True)


# remove 122th column
df= df.drop(122).reset_index(drop=True)

# fix spaces in column name
df.columns=df.columns.str.strip()

 # change the required column as integer data type
df[['month','day','year','Temperature','RH','Ws']]= df[['month','day','year','Temperature','RH','Ws']].astype(int)

# Changing the other columns to Float data type
df[['Rain','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']]=df[['Rain','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)

df.to_csv('Algerian_forest_fires_dataset_Cleaned.csv', index=False)
