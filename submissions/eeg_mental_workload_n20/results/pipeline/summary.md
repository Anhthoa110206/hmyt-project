# Ket qua pipeline STEW

- So nguoi: 48; so file: 96; so epoch giu lai: 6198 (Low 3205, High 2993); epoch bi loai: 906
- Thoi luong moi file: 150.0-150.0 s; so kenh: [14]; gia tri NaN: 0
- Tham so: loc 1-40 Hz; epoch 4 s buoc 2 s; Welch Hann 1 s overlap 50%; bands {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}; relative / (1, 30) Hz
- Danh gia: GroupKFold 8 fold theo nguoi (khong nguoi nao o ca train va test); scaler nam trong Pipeline (fit tren train).

- Diem workload tu danh gia (1-9, n=45): nghi 1.80 ± 1.01, SIMKAP 6.58 ± 1.42

## So sanh mo hinh (bo dac trung: absolute + relative)

| Mo hinh | Balanced acc | Recall High | Precision High | F1 macro |
|---|---|---|---|---|
| Dummy (most frequent) | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.341 ± 0.007 |
| Logistic Regression | 0.803 ± 0.050 | 0.822 ± 0.072 | 0.787 ± 0.071 | 0.801 ± 0.052 |
| KNN (k=15) | 0.744 ± 0.059 | 0.788 ± 0.075 | 0.718 ± 0.073 | 0.742 ± 0.060 |
| SVM (RBF) | 0.788 ± 0.078 | 0.827 ± 0.095 | 0.762 ± 0.089 | 0.785 ± 0.078 |
| Random Forest | 0.802 ± 0.046 | 0.814 ± 0.078 | 0.788 ± 0.056 | 0.801 ± 0.046 |

## Ablation (Logistic Regression)

| Bo dac trung | So dac trung | Balanced acc | Recall High |
|---|---|---|---|
| absolute (log) | 42 | 0.773 ± 0.052 | 0.788 ± 0.099 |
| relative | 42 | 0.773 ± 0.056 | 0.773 ± 0.070 |
| absolute + relative | 84 | 0.803 ± 0.050 | 0.822 ± 0.072 |
| abs + rel + ratios | 87 | 0.803 ± 0.048 | 0.823 ± 0.074 |
| regional (4 features) | 4 | 0.609 ± 0.087 | 0.564 ± 0.144 |
