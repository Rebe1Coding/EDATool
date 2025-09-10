import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math



class Plots:

    def plot_hist_seaborn(self, cols, figsize=(15, 10), bins='auto', kde=True,
                     palette='viridis', **kwargs):
      if cols is None:
          print('Переменные не выбранны')
          return
         

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




    def plot_crosstab_barplot(self, target_col, categorical_cols, figsize=(15, 12), palette='viridis', **kwargs):

      if categorical_cols is None:
          print('Переменные не выбранны')
          return

      

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


    def corr_matrix(self, cols, target):
       if cols is None:
          cols =  self.num_col
       plt.figure(figsize=(10,10))
       sns.heatmap(data=self.df[cols], annot=True, cmap='coolwarm', fmt='.2f')
       plt.show()