import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train.pop('stress_score')
n_train = len(train)
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# 전처리
df['mean_working'] = df['mean_working'].fillna(0)
df = df.fillna('Unknown')
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# get_dummies(drop_first=True)
cat_cols = df.select_dtypes(include=['object']).columns.drop('ID')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
df = df.drop(columns=['ID'])

# Train / Test 재분리
X_train, X_test = df.iloc[:n_train], df.iloc[n_train:]

# 10-Fold 앙상블 학습 및 검증
print("🚀 10-Fold SVR 학습 및 평가 시작...\n")
kf = KFold(n_splits=10, shuffle=True, random_state=777)

oof_preds = np.zeros(len(X_train))  # OOF 점수 계산용
test_preds = np.zeros(len(X_test))  # 최종 제출용

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
    
    # 파이프라인 구성
    model = make_pipeline(
        RobustScaler(),
        TransformedTargetRegressor(
            regressor=SVR(C=3.9635, gamma=1.0631, epsilon=0.0, kernel="rbf"),
            transformer=QuantileTransformer(output_distribution="normal", 
                                            n_quantiles=min(1000, len(y_tr)), random_state=777)
        )
    )
    
    # 모델 학습
    model.fit(X_tr, y_tr)
    
    # 검증 세트 예측 및 점수(MAE) 계산
    val_preds = model.predict(X_va)
    oof_preds[val_idx] = val_preds
    fold_mae = mean_absolute_error(y_va, val_preds)
    
    # 결과 출력
    print(f"Fold {fold+1:02d} MAE: {fold_mae:.5f}")
    
    # Test 예측값 누적 (앙상블)
    test_preds += model.predict(X_test) / 10

# 최종 OOF 점수 출력
total_mae = mean_absolute_error(y, oof_preds)
print(f"\n[최종 결과] 10-Fold CV Total MAE: {total_mae:.5f} \n")

#제출
submit = pd.read_csv("sample_submission.csv")
submit['stress_score'] = np.clip(test_preds, 0, 1) # 0~1 범위 초과 방지
submit.to_csv("minimal_best_submission.csv", index=False)
print("minimal_best_submission.csv 저장됨.")