import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


class lil_vludick_EDA:
    def __init__(self, df:pd.DataFrame):
        self.df = df.copy()
        self.num_col =[col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        self.cat_col = [col for col in self.df.columns if self.df[col].dtype not in self.num_col]

    def show_info(self):
      print(f"Непрерывных переменных: {len(self.num_col)}")
      print(f"Категориальных предикторов:{len(self.cat_col)}")
      desc_df = self.df[self.num_col].describe().T
      print("Описательная статистика")
      print(desc_df.to_markdown())
      print("Описательная статистика для категориальных переменных:")
      print(self.df.describe(include=['category', 'object']).to_markdown())
      print("Кол-во пропущенных")
      print(self.df.isnull().sum())

    def crosstab(self,col1,col2):
        return pd.crosstab(self.df[col1],self.df[col2])




    def plot_hist_seaborn(self, cols = None, figsize=(15, 10), bins='auto', kde=True,
                     palette='viridis', **kwargs):
      if cols is None:
          cols = self.num_col

      n_cols = 3
      n_rows = math.ceil(len(cols) / n_cols)

      fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
      axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

      for i, col in enumerate(cols):
          ax = axes[i]
          sns.histplot(data=self.df, x=col,  ax=ax, **kwargs)
          ax.set_title(f'Распределение {col}', fontsize=12)
          ax.grid(alpha=0.3)

      for j in range(len(cols), len(axes)):
          axes[j].set_visible(False)

      plt.tight_layout()
      plt.show()




    def plot_crosstab_barplot(self, target_col, categorical_cols = None, figsize=(15, 12), palette='viridis', **kwargs):

      if categorical_cols is None:
          categorical_cols = self.cat_col

      if target_col not in self.df.columns:
          raise ValueError(f"Таргет {target_col} отсутствует в DataFrame")

      n_cols = 3
      n_rows = math.ceil(len(categorical_cols) / n_cols)

      fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
      axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

      for i, col in enumerate(categorical_cols):
          ax = axes[i]
          sns.countplot(data=self.df, x=col, hue=target_col, ax=ax, palette=palette, **kwargs)
          ax.set_title(f"Распределение '{col}' по '{target_col}'", fontsize=12)
          ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
          ax.tick_params(axis='x', rotation=45)
          ax.grid(alpha=0.3)

      # Скрываем пустые подграфики
      for j in range(len(categorical_cols), len(axes)):
          axes[j].set_visible(False)

      plt.tight_layout()
      plt.show()









