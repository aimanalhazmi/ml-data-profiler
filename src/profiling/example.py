import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from profiling.empty_detection import empty_detection
from profiling.outlier_detection import OutlierDetector
from profiling.influence_detection import InfluenceOutlierDetector

# === 1. 读取数据 ===
df = pd.read_csv('adult.csv')

# === 2. 清洗数据 ===
# 去除含 '?' 的无效值（处理方式因人而异）
df = df.replace('?', np.nan)
df = df.dropna()

# 选择特征
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# 编码分类变量
df_encoded = pd.get_dummies(df[categorical_cols])
X = pd.concat([df[numerical_cols], df_encoded], axis=1).astype(float)

# 标签编码
y = (df['income'] == '>50K').astype(int).values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 3. Missing Value 检测 ===
print("\n==== Missing Value Detection ====")
mv_report = empty_detection().report(df)
print(mv_report)

# === 4. Outlier 检测 ===
print("\n==== Outlier Detection (Z-score) ====")
outlier_idx = OutlierDetector(method='zscore', threshold=40).detect(X_train)
print("Outlier indices:", np.unique(outlier_idx[0]))

# === 5. influence Function 检测 ===
print("\n==== influence Function Detection ====")
model = LogisticRegression(max_iter=500).fit(X_train, y_train)
infl_detector = InfluenceOutlierDetector(model, X_train, y_train)
topk_idx, infl_scores = infl_detector.detect(X_test[:5], y_test[:5],top_k=50)
print("Top-k influential training points:", np.sort(topk_idx))
print("Top-k influential training score:", infl_scores)
