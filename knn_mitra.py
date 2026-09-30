import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# PENGATURAN TAMPILAN PANDAS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# ============================================================
# 1. PENGUMPULAN DATA
# ============================================================

df = pd.read_csv("dataset_mitra.csv", sep=";")

print("\n=== 1. PENGUMPULAN DATA ===")
print("Jumlah data    :", len(df))
print("Jumlah mitra   :", df["ID Mitra"].nunique())
print("Jumlah periode :", df["Periode"].nunique())


# ============================================================
# 2. SELECTION
# ============================================================

fitur = [
    "Frekuensi Pembelian",
    "Total Pembelian",
    "Lama Bergabung (Bulan)"
]

label = "Level Mitra (Label Kelas)"

X = df[fitur]
y = df[label]

print("\n=== 2. SELECTION ===")
print("Fitur yang digunakan :")
for nama_fitur in fitur:
    print("-", nama_fitur)

print("\nLabel kelas :")
print("-", label)


# ============================================================
# 3. PREPROCESSING
# ============================================================

print("\n=== 3. PREPROCESSING ===")

print("\nJumlah baris :", len(df))
print("Jumlah kolom :", len(df.columns))

print("\nNama kolom:")
print(df.columns.tolist())

print("\nData kosong:")
print(df.isnull().sum().to_string())

print("\nJumlah data duplikat :")
print(df.duplicated().sum())

print("\nDistribusi Level Mitra:")
print(df[label].value_counts().to_string())

print("\nTipe data:")
print(df.dtypes.to_string())


# ============================================================
# 4. TRANSFORMATION
# ============================================================

print("\n=== 4. TRANSFORMATION ===")


# ============================================================
# 4.1 PEMBAGIAN DATA TRAINING DAN TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== 4.1 PEMBAGIAN DATA TRAINING DAN TESTING ===")
print("Total data    :", len(df))
print("Data training :", len(X_train))
print("Data testing  :", len(X_test))

print("\nDistribusi Data Training:")
print(y_train.value_counts().to_string())

print("\nDistribusi Data Testing:")
print(y_test.value_counts().to_string())


# ============================================================
# 4.2 STANDARDISASI DATA
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=fitur,
    index=X_train.index
)

print("\n=== 4.2 STANDARDISASI DATA ===")

print("\n=== 5 DATA TRAINING SEBELUM STANDARDISASI ===")
print(X_train.head().to_string())

print("\n=== 5 DATA TRAINING SETELAH STANDARDISASI ===")
print(X_train_scaled_df.head().to_string())


# ============================================================
# 5. DATA MINING
# ============================================================

print("\n=== 5. DATA MINING ===")


# ============================================================
# 5.1 PENGUJIAN NILAI K DENGAN CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

hasil_k = []

for k in range(1, 21):

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=k
            )
        )
    ])

    skor = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    hasil_k.append({
        "K": k,
        "Accuracy": skor.mean()
    })

hasil_k_df = pd.DataFrame(hasil_k)

print("\n=== 5.1 PENGUJIAN NILAI K DENGAN CROSS-VALIDATION ===")

hasil_k_tampil = hasil_k_df.copy()

hasil_k_tampil["Accuracy"] = (
    hasil_k_tampil["Accuracy"] * 100
).round(2).astype(str) + "%"

print(
    hasil_k_tampil.to_string(
        index=False
    )
)


# ============================================================
# 5.2 PENENTUAN NILAI K TERBAIK
# ============================================================

baris_terbaik = hasil_k_df["Accuracy"].idxmax()

k_terbaik = int(
    hasil_k_df.loc[
        baris_terbaik,
        "K"
    ]
)

akurasi_cv_terbaik = (
    hasil_k_df.loc[
        baris_terbaik,
        "Accuracy"
    ]
)

print("\n=== 5.2 PENENTUAN NILAI K TERBAIK ===")
print("K terbaik             :", k_terbaik)
print(
    f"Rata-rata Accuracy CV : "
    f"{akurasi_cv_terbaik * 100:.2f}%"
)


# ============================================================
# 5.3 PEMBENTUKAN MODEL KNN
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=k_terbaik
)

knn.fit(
    X_train_scaled,
    y_train
)

y_pred = knn.predict(
    X_test_scaled
)

print("\n=== 5.3 PEMBENTUKAN MODEL KNN ===")
print("Nilai K yang digunakan :", k_terbaik)
print("Model KNN berhasil dibentuk.")
print("Prediksi data testing berhasil dilakukan.")


# ============================================================
# 6. INTERPRETATION / EVALUATION
# ============================================================

print("\n=== 6. INTERPRETATION / EVALUATION ===")

labels_kelas = [
    "Agen",
    "Sub-Agen",
    "Reseller"
]


# ============================================================
# 6.1 CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels_kelas
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Aktual Agen",
        "Aktual Sub-Agen",
        "Aktual Reseller"
    ],
    columns=[
        "Prediksi Agen",
        "Prediksi Sub-Agen",
        "Prediksi Reseller"
    ]
)

print("\n=== 6.1 CONFUSION MATRIX ===")
print(cm_df.to_string())


# ============================================================
# 6.2 METRIK EVALUASI
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n=== 6.2 METRIK EVALUASI ===")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# ============================================================
# 7. EVALUASI LEVEL MITRA
# ============================================================

print("\n=== 7. EVALUASI LEVEL MITRA ===")

periode_evaluasi = "Februari 2026"

data_evaluasi = df[
    df["Periode"] == periode_evaluasi
].copy()

X_evaluasi = data_evaluasi[fitur]

# Menggunakan scaler yang telah diperoleh dari data training
X_evaluasi_scaled = scaler.transform(
    X_evaluasi
)

# Menggunakan model KNN yang telah dibentuk
hasil_prediksi = knn.predict(
    X_evaluasi_scaled
)

hasil_evaluasi = pd.DataFrame({
    "ID Mitra":
        data_evaluasi["ID Mitra"].values,

    "Level Mitra Tercatat":
        data_evaluasi[label].values,

    "Hasil KNN":
        hasil_prediksi
})

hasil_evaluasi["Status Evaluasi"] = hasil_evaluasi.apply(
    lambda row:
        "Sesuai"
        if row["Level Mitra Tercatat"] == row["Hasil KNN"]
        else "Perlu Evaluasi",
    axis=1
)

mitra_perlu_evaluasi = hasil_evaluasi[
    hasil_evaluasi["Status Evaluasi"] == "Perlu Evaluasi"
].copy()

jumlah_sesuai = (
    hasil_evaluasi["Status Evaluasi"] == "Sesuai"
).sum()

jumlah_perlu_evaluasi = len(
    mitra_perlu_evaluasi
)

persentase_sesuai = (
    jumlah_sesuai
    / len(hasil_evaluasi)
    * 100
)

persentase_perlu_evaluasi = (
    jumlah_perlu_evaluasi
    / len(hasil_evaluasi)
    * 100
)

print("Periode evaluasi        :", periode_evaluasi)
print("Jumlah mitra dievaluasi :", len(hasil_evaluasi))
print("Jumlah Sesuai           :", jumlah_sesuai)
print("Jumlah Perlu Evaluasi   :", jumlah_perlu_evaluasi)

print(
    f"Persentase Sesuai       : "
    f"{persentase_sesuai:.2f}%"
)

print(
    f"Persentase Perlu Evaluasi : "
    f"{persentase_perlu_evaluasi:.2f}%"
)

print("\nMitra yang Perlu Evaluasi:")
print(
    mitra_perlu_evaluasi.to_string(
        index=False
    )
)


# ============================================================
# 8. PENYIMPANAN HASIL
# ============================================================

hasil_evaluasi.to_csv(
    "hasil_evaluasi_mitra.csv",
    index=False
)

mitra_perlu_evaluasi.to_csv(
    "mitra_perlu_evaluasi.csv",
    index=False
)

hasil_k_df.to_csv(
    "hasil_pengujian_k.csv",
    index=False
)

print("\n=== 8. PENYIMPANAN HASIL ===")
print("hasil_evaluasi_mitra.csv berhasil disimpan.")
print("mitra_perlu_evaluasi.csv berhasil disimpan.")
print("hasil_pengujian_k.csv berhasil disimpan.")


# ============================================================
# 9. VISUALISASI
# ============================================================

print("\n=== 9. VISUALISASI ===")


# ============================================================
# 9.1 GRAFIK PENGUJIAN NILAI K
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    hasil_k_df["K"],
    hasil_k_df["Accuracy"] * 100,
    marker="o"
)

plt.xlabel("Nilai K")
plt.ylabel("Accuracy (%)")
plt.title("Hasil Pengujian Nilai K pada KNN")
plt.xticks(range(1, 21))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "grafik_best_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("grafik_best_k.png berhasil disimpan.")


# ============================================================
# 9.2 CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels_kelas
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    f"Confusion Matrix KNN (K={k_terbaik})"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_knn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("confusion_matrix_knn.png berhasil disimpan.")


# ============================================================
# 9.3 GRAFIK EVALUASI LEVEL MITRA
# ============================================================

jumlah_status = (
    hasil_evaluasi["Status Evaluasi"]
    .value_counts()
    .reindex(
        [
            "Sesuai",
            "Perlu Evaluasi"
        ],
        fill_value=0
    )
)

plt.figure(figsize=(8, 6))

plt.bar(
    jumlah_status.index,
    jumlah_status.values
)

plt.xlabel("Status Evaluasi")
plt.ylabel("Jumlah Mitra")
plt.title(
    "Hasil Evaluasi Kesesuaian Level Mitra"
)

for i, nilai in enumerate(
    jumlah_status.values
):
    plt.text(
        i,
        nilai + 3,
        str(nilai),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "grafik_evaluasi_level.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "grafik_evaluasi_level.png berhasil disimpan."
)


# ============================================================
# 9.4 GRAFIK PERUBAHAN LEVEL MITRA
# ============================================================

mitra_perlu_evaluasi["Perubahan Level"] = (
    mitra_perlu_evaluasi[
        "Level Mitra Tercatat"
    ]
    + " -> "
    + mitra_perlu_evaluasi[
        "Hasil KNN"
    ]
)

jumlah_perubahan = (
    mitra_perlu_evaluasi[
        "Perubahan Level"
    ]
    .value_counts()
)

plt.figure(figsize=(10, 6))

plt.bar(
    jumlah_perubahan.index,
    jumlah_perubahan.values
)

plt.xlabel("Perubahan Level")
plt.ylabel("Jumlah Mitra")
plt.title(
    "Distribusi Perubahan Level Mitra"
)

plt.xticks(rotation=20)

for i, nilai in enumerate(
    jumlah_perubahan.values
):
    plt.text(
        i,
        nilai + 0.2,
        str(nilai),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "grafik_perubahan_level.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "grafik_perubahan_level.png berhasil disimpan."
)


# ============================================================
# 10. SELESAI
# ============================================================

print("\n=== 10. SELURUH PROSES SELESAI ===")
print("Seluruh proses KNN berhasil dijalankan.")