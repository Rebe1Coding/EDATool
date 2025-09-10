import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import spearmanr, pearsonr
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
    
   
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title(f'Матрица корреляций ({method})')
    
    # Визуализация матрицы p-values
        plt.subplot(1, 2, 2,)
    # Создаем маску для значимых корреляций
        mask = pvalue_matrix < 0.05
        sns.heatmap(pvalue_matrix, annot=True, cmap='viridis', 
                fmt='.2f', square=True, cbar_kws={"shrink": .8},
                mask=mask)
        plt.title('Матрица p-values (только значимые p < 0.05)')
    
        plt.tight_layout()
        plt.show()
    
    

    @staticmethod
    def calculate_pvalues(df, method='spearman'):
        df = df.dropna()._get_numeric_data()
        n = len(df.columns)
        pvalues = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    pvalues[i, j] = 0.0
                else:
                    col1 = df.iloc[:, i]
                    col2 = df.iloc[:, j]
                    
                    # Удаляем NaN значения для корректного расчета
                    mask = ~(col1.isna() | col2.isna())
                    col1_clean = col1[mask]
                    col2_clean = col2[mask]
                    
                    if len(col1_clean) > 2: 
                        if method.lower() == 'spearman':
                            _, p_val = spearmanr(col1_clean, col2_clean, nan_policy='omit')
                        else:
                            _, p_val = pearsonr(col1_clean, col2_clean)
                        pvalues[i, j] = p_val
                    else:
                        pvalues[i, j] = np.nan
        
        return pd.DataFrame(pvalues, index=df.columns, columns=df.columns)