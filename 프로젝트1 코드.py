"""
신체 데이터 기반 스트레스 점수 예측 AI 파이프라인
Target: stress_score (0.0 ~ 1.0 연속값 회귀)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("=" * 60)
print("📂 데이터 로드 중...")
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test  = pd.read_csv('/mnt/user-data/uploads/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

# ─────────────────────────────────────────────
# 2. 피처 엔지니어링
# ─────────────────────────────────────────────
print("\n🔧 피처 엔지니어링...")

def feature_engineering(df):
    df = df.copy()

    # BMI
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    # 맥압 (Pulse Pressure)
    df['pulse_pressure'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']

    # 평균 혈압
    df['mean_arterial_pressure'] = df['diastolic_blood_pressure'] + df['pulse_pressure'] / 3

    # 혈압 위험도 (고혈압 기준)
    df['bp_risk'] = ((df['systolic_blood_pressure'] >= 130) | (df['diastolic_blood_pressure'] >= 80)).astype(int)

    # 콜레스테롤 위험도
    df['chol_risk'] = (df['cholesterol'] >= 240).astype(int)

    # 혈당 위험도
    df['glucose_risk'] = (df['glucose'] >= 126).astype(int)

    # 복합 건강 위험 점수
    df['health_risk_score'] = df['bp_risk'] + df['chol_risk'] + df['glucose_risk']

    # 병력 있음/없음 (결측=없음 처리)
    df['has_medical_history'] = df['medical_history'].notna().astype(int)
    df['has_family_history']  = df['family_medical_history'].notna().astype(int)

    # 근무시간 결측 = 0으로 처리 (미취업/무응답)
    df['mean_working_filled'] = df['mean_working'].fillna(0)
    df['is_working'] = (df['mean_working'].notna()).astype(int)

    # 나이 구간
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)

    return df

train = feature_engineering(train)
test  = feature_engineering(test)

# ─────────────────────────────────────────────
# 3. 피처 / 타겟 정의
# ─────────────────────────────────────────────
TARGET = 'stress_score'
DROP_COLS = ['ID', 'stress_score', 'mean_working',
             'medical_history', 'family_medical_history']

NUMERIC_FEATURES = [
    'age', 'height', 'weight', 'cholesterol',
    'systolic_blood_pressure', 'diastolic_blood_pressure',
    'glucose', 'bone_density',
    'bmi', 'pulse_pressure', 'mean_arterial_pressure',
    'health_risk_score', 'mean_working_filled',
    'bp_risk', 'chol_risk', 'glucose_risk',
    'has_medical_history', 'has_family_history',
    'is_working', 'age_group'
]

CATEGORICAL_FEATURES = [
    'gender', 'activity', 'smoke_status',
    'sleep_pattern', 'edu_level'
]

X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = train[TARGET]
X_test_final = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

print(f"  피처 수: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")

# ─────────────────────────────────────────────
# 4. 전처리 파이프라인
# ─────────────────────────────────────────────
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

# ─────────────────────────────────────────────
# 5. 모델 정의
# ─────────────────────────────────────────────
models = {
    'Random Forest': Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ))
    ]),
    'Extra Trees': Pipeline([
        ('pre', preprocessor),
        ('model', ExtraTreesRegressor(
            n_estimators=300, max_depth=12,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('pre', preprocessor),
        ('model', GradientBoostingRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        ))
    ]),
}

# ─────────────────────────────────────────────
# 6. 교차검증 평가
# ─────────────────────────────────────────────
print("\n📊 교차검증 평가 (5-Fold)...")
print("-" * 60)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipeline in models.items():
    rmse_scores = np.sqrt(-cross_val_score(
        pipeline, X, y, cv=cv,
        scoring='neg_mean_squared_error', n_jobs=-1
    ))
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)
    mae_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)

    cv_results[name] = {
        'RMSE': rmse_scores, 'R2': r2_scores, 'MAE': mae_scores
    }
    print(f"\n  [{name}]")
    print(f"    RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"    MAE : {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"    R²  : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

# 최고 모델 선택
best_model_name = min(cv_results, key=lambda k: cv_results[k]['RMSE'].mean())
print(f"\n🏆 최고 모델: {best_model_name}")

# ─────────────────────────────────────────────
# 7. 최종 모델 학습 & 예측
# ─────────────────────────────────────────────
print("\n🚀 전체 데이터로 최종 모델 학습...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)

val_pred = best_pipeline.predict(X_val)
val_pred = np.clip(val_pred, 0, 1)

print(f"\n  ✅ Validation 결과")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")
print(f"    MAE : {mean_absolute_error(y_val, val_pred):.4f}")
print(f"    R²  : {r2_score(y_val, val_pred):.4f}")

# 최종 학습 (전체 데이터)
best_pipeline.fit(X, y)
test_pred = np.clip(best_pipeline.predict(X_test_final), 0, 1)

# ─────────────────────────────────────────────
# 8. 피처 중요도
# ─────────────────────────────────────────────
feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
importances = best_pipeline.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# ─────────────────────────────────────────────
# 9. 시각화
# ─────────────────────────────────────────────
print("\n📈 시각화 생성 중...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f1117')

def style_ax(ax, title):
    ax.set_facecolor('#1a1d27')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#aaaaaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')

ACCENT = ['#7c6fcd', '#56cfb2', '#f4a442', '#e05c7b', '#4ca9e8']

# ── (1) 타겟 분포
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(y, bins=40, color=ACCENT[0], alpha=0.85, edgecolor='#0f1117', linewidth=0.5)
ax1.axvline(y.mean(), color=ACCENT[2], linestyle='--', linewidth=1.5, label=f'Mean={y.mean():.2f}')
ax1.legend(fontsize=9, labelcolor='white', facecolor='#1a1d27', edgecolor='#333344')
style_ax(ax1, 'Stress Score Distribution (Train)')
ax1.set_xlabel('Stress Score')
ax1.set_ylabel('Count')

# ── (2) 예측 분포
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(test_pred, bins=40, color=ACCENT[1], alpha=0.85, edgecolor='#0f1117', linewidth=0.5)
ax2.axvline(test_pred.mean(), color=ACCENT[2], linestyle='--', linewidth=1.5, label=f'Mean={test_pred.mean():.2f}')
ax2.legend(fontsize=9, labelcolor='white', facecolor='#1a1d27', edgecolor='#333344')
style_ax(ax2, 'Predicted Stress Score Distribution (Test)')
ax2.set_xlabel('Predicted Stress Score')
ax2.set_ylabel('Count')

# ── (3) 실제 vs 예측 scatter
ax3 = fig.add_subplot(3, 3, 3)
ax3.scatter(y_val, val_pred, alpha=0.3, s=12, color=ACCENT[0])
ax3.plot([0, 1], [0, 1], color=ACCENT[2], linestyle='--', linewidth=1.5, label='Perfect')
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.legend(fontsize=9, labelcolor='white', facecolor='#1a1d27', edgecolor='#333344')
style_ax(ax3, 'Actual vs Predicted (Validation)')
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')

# ── (4) 피처 중요도 Top 15
ax4 = fig.add_subplot(3, 3, (4, 5))
top15 = feat_imp.head(15)
colors_bar = [ACCENT[0] if i < 5 else ACCENT[1] if i < 10 else ACCENT[2] for i in range(15)]
bars = ax4.barh(top15.index[::-1], top15.values[::-1], color=colors_bar[::-1], height=0.7)
for bar, val in zip(bars, top15.values[::-1]):
    ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left', color='#aaaaaa', fontsize=8)
style_ax(ax4, f'Feature Importance - Top 15 ({best_model_name})')
ax4.set_xlabel('Importance')

# ── (5) 모델별 RMSE 비교
ax5 = fig.add_subplot(3, 3, 6)
model_names = list(cv_results.keys())
rmse_means = [cv_results[m]['RMSE'].mean() for m in model_names]
rmse_stds  = [cv_results[m]['RMSE'].std()  for m in model_names]
bar_colors = [ACCENT[3] if m != best_model_name else ACCENT[1] for m in model_names]
bars2 = ax5.bar(model_names, rmse_means, yerr=rmse_stds, color=bar_colors,
                capsize=6, error_kw={'ecolor': '#aaaaaa', 'linewidth': 1.2})
for bar, val in zip(bars2, rmse_means):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
ax5.set_xticklabels(model_names, rotation=10, ha='right')
style_ax(ax5, 'Model RMSE Comparison (5-Fold CV)')
ax5.set_ylabel('RMSE')

# ── (6) 잔차 분포
ax6 = fig.add_subplot(3, 3, 7)
residuals = y_val.values - val_pred
ax6.hist(residuals, bins=40, color=ACCENT[4], alpha=0.85, edgecolor='#0f1117', linewidth=0.5)
ax6.axvline(0, color=ACCENT[2], linestyle='--', linewidth=1.5)
style_ax(ax6, 'Residuals Distribution (Validation)')
ax6.set_xlabel('Residual (Actual - Predicted)')
ax6.set_ylabel('Count')

# ── (7) 스트레스 레벨별 예측 박스플롯
ax7 = fig.add_subplot(3, 3, 8)
pred_df = pd.DataFrame({'actual': y_val.values, 'predicted': val_pred})
pred_df['actual_bin'] = pd.cut(pred_df['actual'],
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                labels=['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0'])
groups = [pred_df[pred_df['actual_bin'] == b]['predicted'].values
          for b in ['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']]
bp = ax7.boxplot(groups, labels=['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0'],
                 patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], ACCENT):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
style_ax(ax7, 'Prediction by Actual Stress Level')
ax7.set_xlabel('Actual Stress Range')
ax7.set_ylabel('Predicted Score')

# ── (8) R² 비교
ax8 = fig.add_subplot(3, 3, 9)
r2_means = [cv_results[m]['R2'].mean() for m in model_names]
r2_stds  = [cv_results[m]['R2'].std()  for m in model_names]
bar_colors2 = [ACCENT[3] if m != best_model_name else ACCENT[1] for m in model_names]
bars3 = ax8.bar(model_names, r2_means, yerr=r2_stds, color=bar_colors2,
                capsize=6, error_kw={'ecolor': '#aaaaaa', 'linewidth': 1.2})
for bar, val in zip(bars3, r2_means):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9)
ax8.set_xticklabels(model_names, rotation=10, ha='right')
style_ax(ax8, 'Model R² Comparison (5-Fold CV)')
ax8.set_ylabel('R²')

plt.suptitle('Stress Score Prediction — Model Report', color='white',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/stress_report.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  ✅ 시각화 저장 완료")

# ─────────────────────────────────────────────
# 10. 제출 파일 저장
# ─────────────────────────────────────────────
submission = pd.DataFrame({'ID': test['ID'], 'stress_score': test_pred})
submission.to_csv('/mnt/user-data/outputs/submission.csv', index=False)
print(f"\n📁 제출 파일 저장: submission.csv ({len(submission)}행)")
print("\n제출 파일 미리보기:")
print(submission.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("✅ 완료!")
