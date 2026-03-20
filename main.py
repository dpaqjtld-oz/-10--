import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# ==========================================
# 1. 데이터 로드
# ==========================================
TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
SUBMIT_PATH = "sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y = train.pop('stress_score')
n_train = len(train)
df = pd.concat([train, test], axis=0).reset_index(drop=True)

print(f'Train: {n_train}건  |  Test: {len(test)}건')


# ==========================================
# 2. 피처 엔지니어링 (전처리)
# ==========================================
df['mean_working'] = df['mean_working'].fillna(0)
df = df.fillna('Unknown')
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# 노이즈 컬럼 제거
drop_cols = ['mean_working', 'gender']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 체표면적 (Body Surface Area, BSA) — Mosteller 공식
df['bsa'] = np.sqrt(df['height'] * df['weight'] / 3600)

# 콜레스테롤 × 체중 (절대 지질 부담량)
df['chol_weight'] = df['cholesterol'] * df['weight']

# 초과 혈당² (고혈당 위험 제곱)
df['excess_glucose_sq'] = (df['glucose'] - 100).clip(lower=0) ** 2 / 1000

# √(BMI × 혈당) (대사 비만 기하평균)
df['bmi_glucose_geomean'] = np.sqrt(df['bmi'] * df['glucose'])

# 혈당 / 키 (체격 보정 혈당)
df['glucose_per_height'] = df['glucose'] / (df['height'] + 1e-5)

# 혈당^1.5 × BMI (중간 비선형 혈당 × 비만)
df['glucose15_bmi'] = (df['glucose'] ** 1.5) * df['bmi'] / 1000

# 혈당³ × BMI (극단 고혈당 × 비만)
df['glucose3_bmi'] = (df['glucose'] ** 3) * df['bmi'] / 1000000

# 초과혈당² × 콜레스테롤 (고혈당+고지혈 비선형)
df['excess_glucose_sq_chol'] = df['excess_glucose_sq'] * df['cholesterol'] / 100

# excess_glucose_sq × BMI (초과혈당²×비만)
df['excess_glucose_sq_bmi'] = df['excess_glucose_sq'] * df['bmi']

# bmi_glucose_geomean × excess_glucose_sq (기하평균 × 초과혈당²)
df['geomean_excess_sq'] = df['bmi_glucose_geomean'] * df['excess_glucose_sq'] / 10

# glucose15_bmi × 콜레스테롤 (비선형 혈당×비만×지질 3중 결합)
df['glucose15_bmi_chol'] = df['glucose15_bmi'] * df['cholesterol'] / 1000

# 원핫 인코딩
cat_cols = df.select_dtypes(include=['object']).columns.drop('ID')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
df = df.drop(columns=['ID'])

X_train = df.iloc[:n_train].values.astype(np.float64)
X_test  = df.iloc[n_train:].values.astype(np.float64)
y_arr   = y.values

print(f'피처 수: {X_train.shape[1]}개')
print(f'Train: {X_train.shape}, Test: {X_test.shape}\n')


# ==========================================
# 3. Stage 1: SVR 3000-fold
# ==========================================
FOLDS = 3000
kf_svr = KFold(n_splits=FOLDS, shuffle=True, random_state=2024)

svr_pipe = make_pipeline(
    RobustScaler(),
    TransformedTargetRegressor(
        regressor=SVR(C=2.7, gamma=1.0631, epsilon=0.0, kernel='rbf'),
        transformer=QuantileTransformer(
            output_distribution='normal', n_quantiles=3000, random_state=2024)
    )
)

svr_oof        = np.zeros(len(X_train))
svr_test_preds = np.zeros(len(X_test))

print(f"⚙️ [Stage 1] {FOLDS}-Fold SVR 가동 중...")
for fold, (tr_idx, val_idx) in enumerate(kf_svr.split(X_train)):
    m = clone(svr_pipe)
    m.fit(X_train[tr_idx], y_arr[tr_idx])
    svr_oof[val_idx]  = m.predict(X_train[val_idx])
    svr_test_preds   += m.predict(X_test) / FOLDS

    if (fold + 1) % 50 == 0:  # 너무 많이 출력되는 것을 방지하기 위해 50 단위로 출력
        print(f'  SVR Fold {fold+1:04d}/{FOLDS} 완료')

svr_mae = mean_absolute_error(y_arr, svr_oof)
residuals = y_arr - svr_oof   # Stage 2에 전달할 잔차

print(f'\nStage 1 SVR OOF MAE : {svr_mae:.5f}')
print(f'잔차 분포: mean={residuals.mean():.5f}  std={residuals.std():.5f}\n')


# ==========================================
# 4. Stage 2: LightGBM 잔차 모델
# ==========================================
LGB_SEEDS    = [42, 400, 123, 456, 555, 666, 789, 2024, 10223]
LGB_FOLDS    = 700
LGB_PARAMS   = dict(
    objective        = 'regression_l1',   # MAE 직접 최적화
    num_leaves       = 17,                # 단순 트리 (과적합 방지)
    learning_rate    = 0.03,
    n_estimators     = 1000,              # early_stopping으로 실제 횟수 결정
    min_child_samples= 50,                # 리프당 최소 50샘플 (강한 규제)
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbose          = -1,
)

lgb_oof   = np.zeros(len(X_train))
lgb_test  = np.zeros(len(X_test))

print(f'🌲 [Stage 2-A] LGB 잔차 모델: {len(LGB_SEEDS)} seeds × {LGB_FOLDS} fold')

for seed in LGB_SEEDS:
    kf_lgb = KFold(n_splits=LGB_FOLDS, shuffle=True, random_state=seed)

    lgb_oof_s  = np.zeros(len(X_train))
    lgb_test_s = np.zeros(len(X_test))

    for tr_idx, val_idx in kf_lgb.split(X_train):
        model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed)
        model.fit(
            X_train[tr_idx], residuals[tr_idx],
            eval_set=[(X_train[val_idx], residuals[val_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        lgb_oof_s[val_idx] = model.predict(X_train[val_idx])
        lgb_test_s        += model.predict(X_test) / LGB_FOLDS

    seed_mae = mean_absolute_error(residuals, lgb_oof_s)
    print(f'  Seed {seed:5d}: 잔차 MAE={seed_mae:.5f}  (best_iter={model.best_iteration_})')

    lgb_oof  += lgb_oof_s  / len(LGB_SEEDS)
    lgb_test += lgb_test_s / len(LGB_SEEDS)

final_mae = mean_absolute_error(y_arr, svr_oof + lgb_oof)
print(f'  >> 현재 (SVR + LGB) OOF MAE: {final_mae:.5f}\n')


# ==========================================
# 5. Stage 2: XGBoost 잔차 모델 추가
# ==========================================
XGB_SEEDS    = [42, 400, 123, 456, 555, 666, 789, 2024, 10223]
XGB_FOLDS  = 700
XGB_PARAMS = dict(
    objective        = 'reg:absoluteerror',  # MAE 직접 최적화
    max_depth        = 5,
    learning_rate    = 0.03,
    n_estimators     = 1000,
    min_child_weight = 50,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbosity        = 0,
    early_stopping_rounds = 50,
)

xgb_oof  = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

print(f'🌲 [Stage 2-B] XGB 잔차 모델: {len(XGB_SEEDS)} seeds × {XGB_FOLDS} fold')

for seed in XGB_SEEDS:
    kf_xgb = KFold(n_splits=XGB_FOLDS, shuffle=True, random_state=seed)

    xgb_oof_s  = np.zeros(len(X_train))
    xgb_test_s = np.zeros(len(X_test))

    for tr_idx, val_idx in kf_xgb.split(X_train):
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
        model.fit(
            X_train[tr_idx], residuals[tr_idx],
            eval_set=[(X_train[val_idx], residuals[val_idx])],
            verbose=False,
        )
        xgb_oof_s[val_idx] = model.predict(X_train[val_idx])
        xgb_test_s        += model.predict(X_test) / XGB_FOLDS

    seed_mae = mean_absolute_error(residuals, xgb_oof_s)
    best_iter = model.best_iteration if model.best_iteration is not None else model.n_estimators
    print(f'  Seed {seed:5d}: 잔차 MAE={seed_mae:.5f}  (best_iter={best_iter})')

    xgb_oof  += xgb_oof_s  / len(XGB_SEEDS)
    xgb_test += xgb_test_s / len(XGB_SEEDS)


# ==========================================
# 6. 최종 앙상블 및 제출 파일 생성
# ==========================================
# ── LGB + XGB 잔차 앙상블 ──
combined_resid_oof  = (lgb_oof  + xgb_oof)  / 2
combined_resid_test = (lgb_test + xgb_test) / 2

final2_oof  = svr_oof + combined_resid_oof
final2_test = svr_test_preds + combined_resid_test
final2_mae  = mean_absolute_error(y_arr, final2_oof)

print(f'\n{"─"*55}')
print(f'SVR only              : {svr_mae:.5f}')
print(f'SVR + LGB 잔차        : {final_mae:.5f}')
print(f'SVR + (LGB+XGB) 잔차  : {final2_mae:.5f}')
print(f'추가 개선             : {final_mae - final2_mae:+.5f}')
print(f'{"─"*55}\n')

submit = pd.read_csv(SUBMIT_PATH)
submit['stress_score'] = np.clip(final2_test, 0, 1)

SAVE_PATH = "submission_fold3000찐찐막.csv"
submit.to_csv(SAVE_PATH, index=False)

print(f'✅ 저장 완료: {SAVE_PATH}')
print(f'\n최종 결과 요약:')
print(f'  동훈님 원본 (C=3.9635)    : ~0.13007')
print(f'  SVR only (C=3.0)          :  {svr_mae:.5f}')
print(f'  SVR + LGB 잔차            :  {final_mae:.5f}')
print(f'  SVR + (LGB+XGB) 잔차      :  {final2_mae:.5f}')
print(f'\n예측값 분포:')
print(f'  min={final2_test.min():.4f}  max={final2_test.max():.4f}  '
      f'mean={final2_test.mean():.4f}  std={final2_test.std():.4f}')