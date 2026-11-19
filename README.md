# PROYEK KECERDASAN ARTIFISIAL
## KELOMPOK 9

# Smart Delivery Delay Prediction Berbasis Analisis Data Transaksi dan Algoritma Machine Learning

## Anggota Kelompok

| No. | Nama | NIM |
|:---:|---|---|
| 1 | Yanaka Sofia Pardede | 24031554065 | 
| 2 | Desnanty Ratna Aulia | 24031554163 |
| 3 | Annisa Ramadhani | 24031554206 | 

**Program Studi:** S1 Sains Data, Fakultas Matematika dan Ilmu Pengetahuan Alam

**Dosen Pengampu:**
- Yuni Rosita Dewi, M.Si
- Riskyana Dewi Intan Puspitasari, M.Kom.

## Deskripsi Proyek

Proyek ini merupakan proyek mata kuliah **Kecerdasan Artifisial** yang bertujuan untuk memprediksi **risiko keterlambatan pengiriman (late delivery risk)** menggunakan data transaksi dan logistik dari **DataCo Supply Chain Dataset**.

Keterlambatan pengiriman memengaruhi kepuasan pelanggan, efisiensi operasional, dan profitabilitas perusahaan. Melalui pendekatan machine learning, pola keterlambatan yang sulit diamati secara manual dapat diidentifikasi dari data transaksi yang besar dan kompleks.

Proses pengerjaan mencakup Exploratory Data Analysis (EDA), data cleaning, feature engineering, preprocessing menggunakan pipeline, pemodelan klasifikasi dengan **Random Forest**, evaluasi model, analisis feature importance, serta penarikan insight mengenai pola keterlambatan pengiriman.

## Rumusan Masalah

1. Bagaimana membersihkan dan memproses dataset DataCo Supply Chain agar siap digunakan untuk model machine learning (missing value, data sensitif, encoding, dan scaling)?
2. Fitur apa saja yang paling berpengaruh terhadap keterlambatan pengiriman, dan bagaimana feature engineering membantu analisis?
3. Bagaimana performa model Random Forest dalam memprediksi pengiriman yang terlambat?
4. Insight apa yang dapat diperoleh dari hasil evaluasi model untuk memahami pola keterlambatan?

## Sumber Data

| Sumber | Data yang Digunakan | Format Asal |
|---|---|---|
| DataCo Smart Supply Chain (Kaggle / data.world) | Data transaksi, pelanggan, produk, pesanan, pengiriman, dan wilayah | CSV (`DataCoSupplyChainDataset.csv`) |

> Dataset asli **tidak disertakan** dalam repository ini. Unduh dari sumber aslinya dan baca dengan `encoding='latin1'`.

## Cakupan Data

- **Jumlah observasi:** 180.519 baris.
- **Jumlah variabel awal:** 53 kolom.
- **Jumlah kolom setelah cleaning:** 47 kolom (termasuk fitur hasil feature engineering).
- **Variabel target:** `Late_delivery_risk`
- **Jenis masalah:** klasifikasi biner.
- **Domain:** supply chain dan logistik.

| Nilai Target | Keterangan |
|:---:|---|
| 0 | Tidak berisiko terlambat |
| 1 | Berisiko terlambat |

Distribusi target relatif seimbang: **54,83%** kelas 1 (berisiko terlambat) dan **45,17%** kelas 0, sehingga tidak diperlukan penanganan ketidakseimbangan kelas.

## Tahapan Pengerjaan

### 1. Exploratory Data Analysis (EDA)

- Struktur dataset dan pengecekan missing value.
- Distribusi variabel target.
- Distribusi durasi pengiriman aktual (real) dan terjadwal (scheduled).
- Durasi pengiriman dan selisih waktu per kelas risiko.
- Risiko keterlambatan per region, rute, kategori produk, mode pengiriman, jam order, dan hari order.
- Heatmap korelasi fitur numerik dan fitur logistik, serta pair plot.

### 2. Data Cleaning

- Pengecekan missing value: `Customer Lname` (8), `Customer Zipcode` (3), `Order Zipcode` (155.679), dan `Product Description` (180.519, kosong seluruhnya).
- Penghapusan data duplikat (`drop_duplicates`).
- Penghapusan kolom sensitif/tidak relevan: `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer State`, `Customer Zipcode`, `Order Zipcode`.
- Penghapusan baris dengan target kosong.
- Imputasi: median untuk fitur numerik, `"unknown"` untuk fitur kategorikal.

### 3. Feature Engineering

| Fitur | Deskripsi |
|---|---|
| `Time_Difference` | Selisih durasi pengiriman aktual dan terjadwal |
| `Profit_Margin` | `Order Item Profit Ratio` × `Order Item Product Price` |
| `Region_Code` | Kode numerik dari `Order Region` |
| `Route` | Gabungan kota asal pemesanan dan kota pelanggan |
| `Order_Hour` | Jam pemesanan dilakukan |
| `Order_DayOfWeek` | Hari pemesanan (0–6) |

### 4. Data Preprocessing

Fitur dipisahkan menjadi numerik dan kategorikal, lalu diproses dengan `Pipeline` dan `ColumnTransformer`:

- **Numerik:** `SimpleImputer(strategy="median")` → `StandardScaler`
- **Kategorikal:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

**Fitur yang digunakan model (15 fitur):** `Type`, `Days for shipping (real)`, `Days for shipment (scheduled)`, `Benefit per order`, `Sales per customer`, `Order Item Quantity`, `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Profit Ratio`, `Customer City`, `Customer Country`, `Order Region`, `Order State`, `Order Status`.

### 5. Pemodelan Machine Learning

| Parameter | Nilai |
|---|---|
| Algoritma | Random Forest Classifier |
| `n_estimators` | 200 |
| `max_depth` | None |
| `random_state` | 42 |
| Pembagian data | 80% latih, 20% uji (`stratify=y`) |
| Tuning hyperparameter | Tidak dilakukan (parameter standar) |

## Hasil dan Pembahasan

### 1. Hasil EDA dan Insight

| Aspek | Temuan |
|---|---|
| Selisih waktu | Rata-rata `Time_Difference` kelas tepat waktu **-0,71 hari** (lebih cepat dari jadwal), kelas terlambat **+1,62 hari** |
| Durasi aktual | Median durasi pengiriman kelas terlambat (~5 hari) lebih tinggi dari kelas tepat waktu (~3 hari) |
| Korelasi dengan target | `Time_Difference` **0,78**, `Days for shipping (real)` **0,40**, `Days for shipment (scheduled)` **-0,37** |
| Mode pengiriman | First Class memiliki risiko tertinggi (~0,95), diikuti Second Class (~0,77), Same Day (~0,46), dan Standard Class terendah (~0,38) |
| Region | Risiko relatif merata antar region (kisaran ~0,49–0,58) |
| Rute | Seluruh rute teratas bertujuan Caguas, yang merupakan pusat pemesanan terbesar. Risiko tertinggi pada New York City → Caguas dan London → Caguas |
| Kategori produk | Risiko antar kategori hampir serupa, tidak ada kategori yang sangat menonjol |
| Jam order | Risiko ~0,50–0,54 pada pukul 0–11, naik ke ~0,56–0,60 mulai sekitar pukul 12 |
| Hari order | Relatif stabil sepanjang minggu, tidak ada lonjakan berarti |

### 2. Hasil Evaluasi Model

| Metrik | Hasil |
|---|---:|
| Jumlah data uji | 36.104 |
| Accuracy | 1,00 |
| Precision, Recall, F1-score (kelas 0 dan 1) | 1,00 |

**Confusion matrix:** 16.308 data kelas 0 dan 19.796 data kelas 1 seluruhnya terprediksi benar (tanpa kesalahan prediksi).

### 3. Feature Importance

Fitur paling berpengaruh adalah **`Days for shipping (real)`** (~0,47) dan **`Days for shipment (scheduled)`** (~0,32). Fitur lain seperti `Order Status_SUSPECTED_FRAUD`, `Order Status_CANCELED`, serta fitur harga, diskon, dan profit memberikan kontribusi jauh lebih kecil.

### 4. Catatan dan Keterbatasan

- **Akurasi 100% perlu ditafsirkan hati-hati.** Fitur `Days for shipping (real)` dan `Days for shipment (scheduled)` sangat berkaitan langsung dengan definisi keterlambatan (selisih keduanya sudah hampir menentukan nilai target), sehingga model berpotensi mengalami *data leakage*. Fitur durasi aktual baru diketahui setelah pengiriman selesai, sehingga model ini belum mencerminkan prediksi sebelum pengiriman berlangsung.
- Fitur hasil feature engineering (`Time_Difference`, `Profit_Margin`, `Region_Code`, `Route`, `Order_Hour`, `Order_DayOfWeek`) digunakan pada tahap analisis dan insight, tidak termasuk dalam 15 fitur model.
- Tidak dilakukan tuning hyperparameter maupun cross-validation.
- Hasil analisis menunjukkan pola asosiasi, bukan hubungan sebab-akibat.

## Cara Menjalankan Proyek

### 1. Clone Repository

```bash
git clone https://github.com/yanakapardede/prediksi-keterlambatan-pengiriman-dataco.git
cd smart-delivery-delay-prediction-dataco
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

> Catatan: parameter `sparse_output` pada `OneHotEncoder` memerlukan scikit-learn versi 1.2 atau lebih baru.

### 3. Siapkan Dataset

Unduh `DataCoSupplyChainDataset.csv` dari sumber aslinya, lalu sesuaikan `DATA_PATH` di notebook dengan lokasi file Anda.

Jika menggunakan Google Colab, mount Google Drive terlebih dahulu:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Jalankan Notebook

Buka `KODE_PROYEK_KA_KEL_9.ipynb` di Jupyter Notebook atau Google Colab, lalu jalankan sel secara berurutan: load data → EDA → cleaning → feature engineering → preprocessing → pemodelan → evaluasi → feature importance dan insight.
