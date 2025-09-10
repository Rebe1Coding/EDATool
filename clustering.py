import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

class ClusteringModels:
    def __init__(self, df):
        self.df = df
    
    def kmeans_clustering(self, cols, n_clusters=None, plot=True):
        """
        Кластеризация методом K-Means
        """
        if cols is None:
            print("Признаки для кластеризации не указаны")
            return None
        
        # Выбор признаков
        X = self.df[cols]
        
        # Масштабирование данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Определение оптимального числа кластеров (если не задано)
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
            print(f"Оптимальное число кластеров: {n_clusters}")
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Вычисление метрик
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, clusters)
        
        # Вывод метрик
        print("=" * 50)
        print("K-MEANS КЛАСТЕРИЗАЦИЯ - МЕТРИКИ:")
        print("=" * 50)
        print(f"Число кластеров: {n_clusters}")
        print(f"Inertia: {inertia:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Размеры кластеров:")
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        for cluster, size in cluster_sizes.items():
            print(f"  Кластер {cluster}: {size} объектов ({size/len(clusters)*100:.1f}%)")
        
        # Построение графиков
        if plot:
            self._plot_clusters(X_scaled, clusters, cols, kmeans.cluster_centers_)
        
        # Создание Series с результатами кластеризации
        cluster_series = pd.Series(clusters, index=self.df.index, name='cluster')
        
        return cluster_series, {
            'n_clusters': n_clusters,
            'inertia': inertia,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def _find_optimal_clusters(self, X_scaled, max_clusters=10):
        """
        Поиск оптимального числа кластеров методом локтя и силуэта
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            if k > 1:  # Silhouette score требует хотя бы 2 кластера
                silhouette_scores.append(silhouette_score(X_scaled, clusters))
        
        # Метод локтя
        differences = np.diff(inertias)
        differences_ratio = differences[1:] / differences[:-1]
        optimal_k_elbow = np.argmin(differences_ratio) + 3  # +2 потому что начинаем с k=2
        
        # Метод силуэта
        if silhouette_scores:
            optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        
        # Выбираем оптимальное k (предпочтение силуэту)
        optimal_k = optimal_k_silhouette if silhouette_scores else optimal_k_elbow
        
        # Построение графиков для выбора числа кластеров
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График метода локтя
        ax1.plot(range(2, max_clusters + 1), inertias, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
        
        # График силуэтных score
        if silhouette_scores:
            ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'go-')
            ax2.set_xlabel('Number of clusters')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Method')
            ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        return optimal_k
    
    def _plot_clusters(self, X_scaled, clusters, feature_names, centers):
        """
        Построение графиков кластеризации
        """
        # PCA для визуализации в 2D (если больше 2 признаков)
        if X_scaled.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            centers_2d = pca.transform(centers)
            x_label, y_label = 'PCA Component 1', 'PCA Component 2'
        else:
            X_2d = X_scaled
            centers_2d = centers
            x_label, y_label = feature_names[0], feature_names[1]
        
        # Создание subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot кластеров
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', 
                             alpha=0.6, s=50)
        ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', 
                   s=200, label='Centroids')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title('Кластеризация K-Means')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавление цветовой палитры
        plt.colorbar(scatter, ax=ax1)
        
        # Bar plot размеров кластеров
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        ax2.bar(cluster_sizes.index.astype(str), cluster_sizes.values, 
               color=plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes))))
        ax2.set_xlabel('Кластер')
        ax2.set_ylabel('Количество объектов')
        ax2.set_title('Размеры кластеров')
        
        # Добавление значений на столбцы
        for i, v in enumerate(cluster_sizes.values):
            ax2.text(i, v + 0.01 * max(cluster_sizes.values), str(v), 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Heatmap центроидов кластеров
        if len(feature_names) > 1:
            plt.figure(figsize=(10, 6))
            centers_df = pd.DataFrame(centers, columns=feature_names, 
                                    index=[f'Cluster {i}' for i in range(len(centers))])
            sns.heatmap(centers_df, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', linewidths=0.5)
            plt.title('Центроиды кластеров (стандартизированные значения)')
            plt.show()

# Пример использования:
# clustering = ClusteringModels(your_dataframe)
# 
# Для кластеризации с автоматическим подбором кластеров:
# clusters, metrics = clustering.kmeans_clustering(['feature1', 'feature2', 'feature3'])
#
# Для кластеризации с заданным числом кластеров:
# clusters, metrics = clustering.kmeans_clustering(['feature1', 'feature2'], n_clusters=3)
#
# clusters будет содержать Series с номерами кластеров для каждого наблюдения