import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns


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
    "energy",
    "loudness",
    # "key",
    "mode",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    # "liveness",
    # "time_signature",
    "duration_min",
]


## 設定 X, y
X = genre_data_unique[columns_for_clustering]
y = genre_data_unique["genre"]


## 處理類別不平衡問題
# print("各類別分布：")
# print(y.value_counts())
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# print("SMOTE 處理後各類別分布：")
# print(pd.Series(y_resampled).value_counts())


## Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled
)


## 建立 Random Forest 模型
svc = SVC(random_state=42)

param_grid = {
    'C': [1, 10],
    'kernel': ['linear'],
}

grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_) 



## 使用最佳參數建立模型
best_params = grid_search.best_params_

best_svc = SVC(
    **best_params,
    random_state=42
)
best_svc.fit(X_train, y_train)

y_pred = best_svc.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_acc)


## 評估效果
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_svm.png")  # 另存圖片
plt.close()
print(classification_report(y_test, y_pred))


## Feature Importance
# SVC 在多分類問題下，coef_ 會是 [n_classes, n_features]
coefficients = best_svc.coef_

# 若是 One-vs-Rest，多個 class 可能各自有一組係數；可依需求合併或分別觀察
# 這裡示範簡單將每個特徵在各類別的絕對值平均作為重要度
abs_coeff_mean = np.mean(np.abs(coefficients), axis=0)

feature_importance_df = pd.DataFrame({
    'Feature': columns_for_clustering,
    'Importance': abs_coeff_mean
}).sort_values('Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(8, 6))
sns.barplot(
    data=feature_importance_df, 
    x='Importance', 
    y='Feature',
    color='skyblue'
)
plt.title("Feature Importance (Linear SVM Coefficients)")
plt.savefig("Feature_Importance_SVM.png")
plt.close()
