import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


class Data:
    
    def __init__(self, df:pd.DataFrame):
        self.__df = df.copy()
        self.num_col =[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        self.cat_col = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]

    def show_info(self):
      print(f"Непрерывных переменных: {len(self.num_col)}")
      print(f"Категориальных предикторов:{len(self.cat_col)}")
      desc_df = self.df[self.num_col].describe().T
      print("Описательная статистика")
      print(desc_df.to_markdown())
      print("Описательная статистика для категориальных переменных:")
      if len(self.cat_col)>0:
        print(self.df.describe(include=['category', 'object']).to_markdown())
      print("Кол-во пропущенных")
      print(self.df.isnull().sum())

    def crosstab(self,col1,col2):
        return pd.crosstab(self.df[col1],self.df[col2])

    @property 
    def df(self):
       return self.__df
    
    @df.setter
    def df(self, new_df:pd.DataFrame):
       self.__df = new_df
       
    def __str__(self):
        print(self.df.head(5).to_markdown())
    


    



        

       
       







