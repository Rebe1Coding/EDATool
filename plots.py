import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import spearmanr
sns.set_theme()



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


    def corr_matrix(self, df, cols, target=None, method='spearman'):
        
    
  
        data = df[cols]
    

        corr_matrix = data.corr(method = method.lower())
        pvalue_matrix = self.calculate_pvalues(data, method=method)
    
   
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title(f'Матрица корреляций ({method})')
    
    # Визуализация матрицы p-values
        plt.subplot(1, 2, 2)
    # Создаем маску для значимых корреляций
        mask = pvalue_matrix > 0.05
        sns.heatmap(pvalue_matrix, annot=True, cmap='viridis', 
                fmt='.3f', square=True, cbar_kws={"shrink": .8},
                mask=mask)
        plt.title('Матрица p-values (только значимые p < 0.05)')
    
        plt.tight_layout()
        plt.show()
    
    

    @staticmethod
    def calculate_pvalues(df, method='spearman'):
        df = df.dropna()._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                if method == 'spearman':
                    corr, p_val = spearmanr(df[r], df[c])
                else:
                    from scipy.stats import pearsonr
                    corr, p_val = pearsonr(df[r], df[c])
                pvalues[r][c] = p_val
        return pvalues