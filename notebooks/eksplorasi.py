import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/kredit_sumatera.csv")

print("--- Q1: tingkat gagal bayar ---")
print(f"{df.gagal_bayar.mean():.2%}")

print("\n--- Q2: komposisi segmen per tahun ---")
print(pd.crosstab(df.periode.str[:4], df.segmen, normalize="index").round(3))

print("\n--- Q3: nilai hilang ---")
hilang = df.isna().mean().sort_values(ascending=False)
print(hilang[hilang > 0].round(4))
print(df.groupby("segmen")["rata_saldo_90h"].apply(lambda s: s.isna().mean()).round(3))

print("\n--- Q4: kolom mencurigakan ---")
print(df.groupby("gagal_bayar")["status_restrukturisasi"].apply(lambda s: s.notna().mean()))

df["rasio_angsuran"] = df.angsuran_bulanan / df.penghasilan_bulanan
num = ["usia", "tanggungan", "lama_usaha_tahun", "penghasilan_bulanan",
       "ragam_penghasilan", "plafon_diajukan", "tenor_bulan", "bunga_tahunan",
       "rasio_angsuran", "rata_saldo_90h", "jumlah_tunggakan_12b"]
X = df[num].join(pd.get_dummies(df[["segmen", "wilayah", "kanal"]]))
y = df.gagal_bayar

latih = df.periode <= "2023-12"
uji = df.periode > "2024-06"
m = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06,
                                   max_iter=250, random_state=42)
m.fit(X[latih], y[latih])
auc_oot = roc_auc_score(y[uji], m.predict_proba(X[uji])[:, 1])

Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
m2 = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06,
                                    max_iter=250, random_state=42).fit(Xa, ya)
auc_acak = roc_auc_score(yb, m2.predict_proba(Xb)[:, 1])

print("\n==========================================")
print(f"AUC out-of-time    : {auc_oot:.4f}")
print(f"AUC pembagian acak : {auc_acak:.4f}")
print("==========================================")
