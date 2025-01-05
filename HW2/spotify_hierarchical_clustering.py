import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn.metrics import (
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    confusion_matrix,
)
from scipy.cluster.hierarchy import dendrogram, linkage


def analyze_column_stats(dataframe, column_name):
    """
    接受一個DataFrame，列印出特定欄位的最大值、最小值，以及NaN總量。

    參數：
    dataframe (pd.DataFrame): 要分析的資料框
    column_name (str): 欲分析的欄位名稱
    """
    try:
        # column_name in dataframe.columns:
        max_value = dataframe[column_name].max()
        min_value = dataframe[column_name].min()
        nan_count = dataframe[column_name].isna().sum()

        print(f"欄位 '{column_name}' 的最大值為: {max_value}")
        print(f"欄位 '{column_name}' 的最小值為: {min_value}")
        print(f"欄位 '{column_name}' 的 NaN 總量為: {nan_count}")
    except:
        print(f"欄位 '{column_name}' 在資料框中不存在！或純文字資料！")


def plot_column_distributions(dataframe, columns):
    """
    接受一個DataFrame和欄位列表，繪製每個欄位的數值分布圖。

    參數：
    dataframe (pd.DataFrame): 要分析的資料框
    columns (list of str): 欲繪製的欄位名稱列表
    """
    for column in columns:
        if column in dataframe.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(dataframe[column], kde=True)
            plt.title(f"{column} 的數值分布圖")
            plt.xlabel(column)
            plt.ylabel("頻率")
            plt.show()
        else:
            print(f"欄位 '{column}' 在資料框中不存在！")


def plot_elbow_diagram(dataframe, columns, max_clusters=20):
    """
    使用Agglomerative Clustering繪製Elbow Diagram以評估最佳的群數。
    Agglomerative Clustering沒有SSE的概念，因此這裡通過樹狀圖(dendrogram)來幫助選擇群數。

    參數：
    dataframe (pd.DataFrame): 要分析的資料框
    columns (list of str): 欲進行Hierarchical Clustering的欄位名稱列表
    max_clusters (int): 最大的聚類數量，用於繪製樹狀圖
    """
    scaled_data = dataframe[columns]

    # 使用 linkage 方法計算距離並繪製樹狀圖
    linked = linkage(scaled_data, method="ward")

    # 繪製樹狀圖
    plt.figure(figsize=(10, 6))
    dendrogram(
        linked,
        truncate_mode="lastp",
        p=max_clusters,
        leaf_rotation=45.0,
        leaf_font_size=12.0,
        show_contracted=True,
    )
    plt.xlabel("Number of Points in Node (or Index of Point)")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()


def hierarchical_clustering_analysis(dataframe, columns, optimal_clusters):
    """
    接受一個DataFrame和欄位列表，並使用Agglomerative Clustering算法進行群聚分析，輸出聚類參數、群數、每群的數量以及主要音樂類別。

    參數：
    dataframe (pd.DataFrame): 要分析的資料框
    columns (list of str): 欲進行Hierarchical Clustering的欄位名稱列表
    optimal_clusters (int): 使用的最佳聚類數量
    """
    scaled_data = dataframe[columns]

    # 使用最佳群數進行 Agglomerative Clustering 聚類
    agglomerative = AgglomerativeClustering(
        n_clusters=optimal_clusters, metric="euclidean", linkage="ward"
    )
    dataframe["cluster"] = agglomerative.fit_predict(scaled_data)

    # 列印每個群的數量和主要音樂類別（假設有一個叫做 'genre' 的欄位表示音樂類別）
    cluster_counts = dataframe["cluster"].value_counts()
    print("每個群的數量:")
    print(cluster_counts)

    if "genre" in dataframe.columns:
        for cluster in range(optimal_clusters):
            major_genre = dataframe[dataframe["cluster"] == cluster]["genre"].mode()[0]
            print(f"群 {cluster} 的主要音樂類別: {major_genre}")
    else:
        print("資料框中沒有 'genre' 欄位，無法計算主要音樂類別。")

    # 評估聚類結果
    if "genre" in dataframe.columns:
        true_labels = (
            dataframe["genre"].astype("category").cat.codes
        )  # 將 genre 編碼為數值類別
        predicted_labels = dataframe["cluster"]

        # Rand Index
        rand_index = rand_score(true_labels, predicted_labels)
        print(f"Rand Index: {rand_index}")

        # Normalized Mutual Information
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        # Adjusted Mutual Information
        ami = adjusted_mutual_info_score(true_labels, predicted_labels)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # V-measure
        v_measure = v_measure_score(true_labels, predicted_labels)
        print(f"V-measure: {v_measure}")

        # Fowlkes-Mallows Scores
        fms = fowlkes_mallows_score(true_labels, predicted_labels)
        print(f"Fowlkes-Mallows Scores: {fms}")

        # Silhouette Coefficient
        silhouette = silhouette_score(scaled_data, predicted_labels)
        print(f"Silhouette Coefficient: {silhouette}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("資料框中沒有 'genre' 欄位，無法計算聚類評估指標。")


## !!! Start !!! ##
## read data
genre_data = pd.read_csv("genres_v2.csv", low_memory=False)
# print("Shape of genre_data:")
# print("Number of rows:", genre_data.shape[0])
# print("Number of columns:", genre_data.shape[1])

## drop columns and duplicates
genre_data = genre_data.drop(
    columns=["title", "Unnamed: 0", "uri", "track_href", "analysis_url"]
)
genre_data = genre_data.drop_duplicates()

## delete duplicate id
genre_data_unique = genre_data.drop_duplicates(subset="id", keep="first")
# print("Shape of genre_data_unique:")
# print("Number of rows:", genre_data_unique.shape[0])
# print("Number of columns:", genre_data_unique.shape[1])

genre_data_unique["genre"].unique()

## 欄位值域分析
# for column in genre_data_unique.columns:
#     print(f"分析欄位 '{column}' 的統計資訊：")
#     analyze_column_stats(genre_data_unique, column)

## 修改欄位
genre_data_unique["duration_min"] = genre_data_unique["duration_ms"] / 60000
genre_data_unique.drop("duration_ms", axis=1, inplace=True)

## 減少偏態
columns_to_transform = ["speechiness", "acousticness", "instrumentalness", "liveness"]
pt = PowerTransformer(method="yeo-johnson")
genre_data_unique[columns_to_transform] = pt.fit_transform(
    genre_data_unique[columns_to_transform]
)
# plot_column_distributions(genre_data_unique, columns_to_transform)


## 標準化
columns_to_transform = [
    "danceability",
    "energy",
    "loudness",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "duration_min",
]
scaler = StandardScaler()
genre_data_unique[columns_to_transform] = scaler.fit_transform(
    genre_data_unique[columns_to_transform]
)
# plot_column_distributions(genre_data_unique, columns_to_transform)


## 執行 KMeans elbow_diagram
columns_for_clustering = [
    "danceability",
    "instrumentalness",
    "tempo",
    "duration_min",
    "loudness",
    # "energy",
    # "key",
    # "mode",
    # "valence",
    # "speechiness",
    # "acousticness",
    # "liveness",
    # "time_signature",
]
# plot_elbow_diagram(genre_data_unique, columns_for_clustering)


## 執行 KMeans 群聚分析
optimal_clusters = 14
hierarchical_clustering_analysis(
    genre_data_unique, columns_for_clustering, optimal_clusters
)
