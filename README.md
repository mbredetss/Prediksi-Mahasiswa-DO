# **Laporan Proyek Machine Learning - Mighdad Abdul Fattah Jaba**

## **Domain Proyek**
Sebagai mahasiswa yang saat ini berada di semester 6, saya merasa sedih karena pada setiap pergantian semester, beberapa teman harus keluar dari kampus atau tidak lulus mata kuliah. Oleh karena itu, saya ingin mengembangkan model prediksi yang dapat memperkirakan hasil akhir nilai ujian mahasiswa. Dengan demikian, mahasiswa yang diprediksi tidak lulus dapat mendapatkan tindakan intervensi sejak dini.

Berdasarkan artikel dari [Departement of Communications UII](https://communication.uii.ac.id/beragam-alasan-mahasiswa-tidak-lulus-lulus-bagaimana-mengantisipasinya/) kasus DO di indonesia ada 601.333 mahasiswa putus kuliah.

Harapannya, langkah ini dapat mengurangi jumlah mahasiswa yang harus mengulang mata kuliah di semester berikutnya. Model ini akan menggunakan faktor-faktor yang memengaruhi nilai ujian sebagai dasar analisis untuk memprediksi performa akademik mahasiswa.

## **Business Understanding**

### **Problem Statement**
Setiap semester, ratusan ribu mahasiswa di Indonesia mengalami kegagalan dalam menyelesaikan mata kuliah atau bahkan mengalami putus kuliah (Drop Out/DO). Salah satu penyebab utama dari permasalahan ini adalah kurangnya sistem yang mampu mengidentifikasi mahasiswa dengan risiko akademik tinggi secara real-time dan memberikan intervensi dini. Saat ini, kampus umumnya hanya melakukan evaluasi setelah ujian akhir, yang menyebabkan mahasiswa baru menyadari risiko akademiknya setelah terlambat untuk melakukan perbaikan.

Permasalahan ini berkontribusi terhadap peningkatan beban akademik dan finansial bagi mahasiswa, serta menurunkan motivasi belajar mereka. Oleh karena itu, diperlukan solusi yang dapat membantu perguruan tinggi dalam memantau dan memprediksi performa akademik mahasiswa sebelum terlambat.

### **Goals**
Untuk mengatasi permasalahan ini, proyek ini bertujuan untuk:

1. Mengembangkan model machine learning yang dapat memprediksi hasil akhir nilai ujian mahasiswa dengan akurasi tinggi berdasarkan faktor-faktor yang memengaruhi performa akademik.
2. Mengidentifikasi faktor dominan yang berkorelasi signifikan dengan ketidaklulusan mahasiswa, seperti tingkat kehadiran, nilai tugas, dan partisipasi kelas.
3. Memberikan rekomendasi berbasis data untuk membantu pihak akademik dalam mengidentifikasi mahasiswa berisiko dan memberikan intervensi lebih awal.

#### **Solution Statement**
Solusi yang diusulkan adalah membangun sistem prediksi berbasis machine learning yang dapat membantu institusi pendidikan dalam mengidentifikasi mahasiswa dengan risiko akademik tinggi. Model ini akan memanfaatkan berbagai fitur seperti data kehadiran, nilai tugas, partisipasi kelas, dan faktor lain yang relevan. Dengan menggunakan pendekatan ini, universitas dapat menerapkan intervensi akademik lebih dini guna meningkatkan tingkat kelulusan mahasiswa.

Karena yang ingin diprediksi adalah nilai ujian akhir (variabel kontinu), maka pendekatan yang digunakan adalah model regresi. Model yang akan dicoba meliputi:

- **Linear Regression**: Model dasar yang mudah diinterpretasi dan cocok sebagai baseline.
- **Decision Tree Regressor**: Dapat menangani hubungan non-linear serta interaksi antar fitur, meskipun memiliki risiko overfitting.
- **Random Forest Regressor**: Mengurangi risiko overfitting yang ada pada Decision Tree dengan menggabungkan beberapa pohon keputusan.

Untuk mengukur performa model yang dikembangkan, metrik yang akan digunakan adalah:

- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.
- **Mean Squared Error (MSE)**: Memberikan penalti lebih besar pada kesalahan yang lebih besar, membantu mendeteksi prediksi dengan error besar.

Dengan pendekatan ini, diharapkan model dapat membantu universitas dalam memantau performa akademik mahasiswa secara lebih efektif dan memberikan intervensi yang tepat waktu.



## **Data Understanding**

Dataset ini memberikan gambaran menyeluruh tentang berbagai faktor yang mempengaruhi kinerja siswa dalam ujian. Data ini mencakup informasi mengenai kebiasaan belajar, kehadiran, keterlibatan orang tua, dan aspek-aspek lain yang memengaruhi keberhasilan akademik. Sumber dataset: [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors). Dataset ini terdiri dari 20 fitur dan 6607 baris. Dataset ini memiliki beberapa data yang hilang. Karena jumlahnya tidak terlalu banyak, data hilang tersebut langsung dihapus. Selanjutnya, terdapat beberapa outlier dalam dataset yang ditangani dengan mengganti nilai ekstrem menggunakan batas atas/bawah (boundary). Tidak ditemukan data tidak valid dalam dataset ini (misalnya, nilai 0 pada kolom 'jam belajar' yang tidak logis). Selain itu, dataset ini memiliki rentang skala fitur yang beragam, tidak memiliki duplikasi data, dan tidak terdapat fitur dengan tingkat multikolinearitas tinggi.

### **Informasi Dataset**
Dataset terdiri dari berbagai variabel yang menggambarkan faktor-faktor yang dapat mempengaruhi nilai ujian siswa. Berikut adalah variabel-variabel dalam dataset ini:

* **Hours_Studied**: Jumlah jam belajar per minggu.
* **Attendance**: Persentase kehadiran di kelas.
* **Parental_Involvement**: Tingkat keterlibatan orang tua dalam pendidikan siswa (Rendah, Sedang, Tinggi).
* **Access_to_Resources**: Ketersediaan sumber daya pendidikan (Rendah, Sedang, Tinggi).
* **Extracurricular_Activities**: Partisipasi dalam kegiatan ekstrakurikuler (Ya, Tidak).
* **Sleep_Hours**: Rata-rata jam tidur per malam.
* **Previous_Scores**: Nilai dari ujian sebelumnya.
* **Motivation_Level**: Tingkat motivasi siswa (Rendah, Sedang, Tinggi).
* **Internet_Access**: Ketersediaan akses internet (Ya, Tidak).
* **Tutoring_Sessions**: Jumlah sesi bimbingan belajar yang dihadiri per bulan.
* **Family_Income**: Tingkat pendapatan keluarga (Rendah, Sedang, Tinggi).
* **Teacher_Quality**: Kualitas guru (Rendah, Sedang, Tinggi).
* **School_Type**: Jenis sekolah yang dihadiri (Negeri, Swasta).
* **Peer_Influence**: Pengaruh teman sebaya terhadap performa akademik (Positif, Netral, Negatif).
* **Physical_Activity**: Rata-rata jam aktivitas fisik per minggu.
* **Learning_Disabilities**: Adanya gangguan belajar (Ya, Tidak).
* **Parental_Education_Level**: Tingkat pendidikan tertinggi orang tua (SMA, Perguruan Tinggi, Pascasarjana).
* **Distance_from_Home**: Jarak dari rumah ke sekolah (Dekat, Sedang, Jauh).
* **Gender**: Jenis kelamin siswa (Laki-laki, Perempuan).
* **Exam_Score**: Nilai ujian akhir.


## **Data Preparation**

Tahap ini mencakup proses preprocessing data agar siap digunakan untuk analisis lebih lanjut. Beberapa langkah yang dilakukan meliputi penanganan missing values, validasi data, deteksi dan penanganan outlier, feature encoding, pembagian dataset (data splitting), feature scaling, serta analisis dan reduksi dimensi menggunakan PCA.

### **1. Handling Missing Values**
Sebelum melakukan analisis lebih lanjut, langkah pertama adalah mengecek apakah terdapat data yang hilang (missing values) dalam dataset. Jika ditemukan, dilakukan penanganan sesuai dengan karakteristik masing-masing fitur, seperti mengisi dengan mean/median (untuk fitur numerik) atau modus (untuk fitur kategorikal).

```python
# 1. Cek Missing Value
print("Missing Values:")
print(StudentPerformanceFactors.isnull().sum())
```

### **2. Data Validation (Pengecekan Data Tidak Valid)**
Setelah menangani missing values, dilakukan pengecekan terhadap data yang tidak valid dengan membatasi nilai dalam rentang yang masuk akal.

```python
valid_ranges = {
    'Hours_Studied': (0, 168),  # Maksimal 24x7 jam
    'Attendance': (0, 100),     # Persentase
    'Sleep_Hours': (0, 24),     # Jam tidur harian
    'Previous_Scores': (0, 100),# Nilai ujian
    'Tutoring_Sessions': (0, 31),# Maksimal 1x sehari
    'Physical_Activity': (0, 168) # Jam per minggu
}
```

### **3. Handling Outliers**
Deteksi outlier dilakukan menggunakan boxplot, dan penanganan dilakukan dengan metode Interquartile Range (IQR) dan mengganti outlier dengan nilai boundary.

```python
# 4. Penanganan Outlier
def handle_outliers(df, column):
    q1 = StudentPerformanceFactors[column].quantile(0.25)
    q3 = StudentPerformanceFactors[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Ganti outlier dengan nilai boundary
    StudentPerformanceFactors[column] = StudentPerformanceFactors[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Terapkan ke semua kolom numerik
for col in numeric_columns:
    df = handle_outliers(StudentPerformanceFactors, col)
```

### **4. Menghapus Fitur yang Memiliki korelasi yang Rendah Dengan Fitur Exam_Score**
Berdasarkan hasil analisis korelasi, fitur Sleep_Hours dan Physical_Activity memiliki skor korelasi yang sangat kecil, sehingga fitur-fitur tersebut dihapus dari dataset.
```python
StudentPerformanceFactors.drop(['Sleep_Hours', 'Physical_Activity'], inplace=True, axis=1)
```

### **5. Memeriksa Data Duplikat**
Seperti yang sudah disampaikan pada bagian data understanding, dataset ini tidak memiliki data duplikat.

### **6. Feature Encoding**
Beberapa fitur dalam dataset memiliki tipe kategorikal, sehingga perlu dikonversi ke bentuk numerik menggunakan Label Encoding agar dapat diproses oleh model machine learning.

```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Parental_Involvement', 'Extracurricular_Activities', 'Motivation_Level',
                   'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type',
                   'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
                   'Distance_from_Home', 'Gender', 'Access_to_Resources']

# Salin dataset ke variabel baru untuk menghindari modifikasi dataset asli
df_encoded = StudentPerformanceFactors.copy()

# Inisialisasi LabelEncoder
le = LabelEncoder()

# Lakukan label encoding untuk setiap kolom kategorikal
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
```

### **7. Data Splitting (Pembagian Dataset)**
Dataset dibagi menjadi data latih (training data) dan data uji (testing data) dengan rasio 80:20.

```python
from sklearn.model_selection import train_test_split
# Pisahkan fitur dan target
X = df_encoded.drop('Exam_Score', axis=1)
y = df_encoded['Exam_Score']
# Split data dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **8. Feature Scaling**
Feature scaling dilakukan setelah data dibagi agar menghindari data leakage. StandardScaler digunakan untuk menstandarkan fitur agar memiliki mean = 0 dan standar deviasi = 1.

```python
from sklearn.preprocessing import StandardScaler
# Scaling pada fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fitting hanya pada data training
X_test_scaled = scaler.transform(X_test)        # Transform data testing
```

### **9. Multicollinearity Check & Principal Component Analysis (PCA)**
Sebelum menerapkan PCA, dilakukan pengecekan multikolinearitas menggunakan Variance Inflation Factor (VIF). Jika terdapat fitur dengan VIF > 10, maka dilakukan reduksi dimensi menggunakan PCA.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Hitung VIF untuk setiap fitur
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

print("VIF sebelum PCA:")
print(vif_data)
```

Jika ditemukan fitur dengan VIF > 10, maka dilakukan reduksi dimensi menggunakan PCA.

```python
from sklearn.decomposition import PCA
# Handle multikolinearitas dengan PCA jika VIF > 10
if any(vif_data["VIF"] > 10):
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"\nMenggunakan {pca.n_components_} komponen PCA.")
    print("Varians dipertahankan:", sum(pca.explained_variance_ratio_))
else:
    print("\nTidak ada multikolinearitas tinggi.")
```

Namun, pada kasus ini, tidak ditemukan multikolinearitas tinggi sehingga PCA tidak diterapkan.


## **Modelling**
Proses selanjutnya merupakan tahap pemodelan machine learning. Pada tahap ini, dikembangkan tiga model berbasis algoritma berbeda untuk kemudian dievaluasi performanya. Tujuan evaluasi adalah membandingkan akurasi prediksi masing-masing model dan menentukan algoritma dengan kinerja optimal. Algoritma yang digunakan beserta cara kerjanya meliputi:
### 1. Linear Regression
Linear regression adalah **model parametrik** yang memodelkan hubungan linear antara fitur (variabel independen) dan target (variabel dependen).
- Persamaan matematis:  
$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
$$
 
 di mana:  
  - $y$: Variabel target  
  - $\beta_0$: Intercept  
  - $\beta_1, ..., \beta_n$: Koefisien fitur  
  - $\epsilon$: Error term  

#### Cara Kerja
1. **Optimasi**: Meminimalkan *Sum of Squared Residuals (SSR)* menggunakan metode **Ordinary Least Squares (OLS)**.  
2. **Asumsi**:  
   - Linearitas antara fitur dan target.  
   - Residual berdistribusi normal dan homoskedastis.  
   - Tidak ada multikolinearitas tinggi antar-fitur. 

### 2. Decision Tree Regressor
Decisoin tree regresor merupakan **model non-parametrik** berbasis struktur pohon keputusan yang membagi data secara rekursif berdasarkan kriteria pemecahan (*split*) untuk meminimalkan varians dalam subset hasil.  

#### Cara Kerja
1. **Pemecahan (*Splitting*)**:  
   - Memilih fitur dan threshold yang memaksimalkan reduksi *Mean Squared Error (MSE)*.  
   - Kriteria pemecahan: $ \text{MSE}_{\text{parent}} - (\text{MSE}_{\text{left}} + \text{MSE}_{\text{right}}) $.  
2. **Penghentian (*Stopping Criteria*)**:  
   - Berhenti jika mencapai `max_depth` atau `min_samples_split`.  
3. **Prediksi**: Nilai target pada *leaf node* adalah rata-rata sampel dalam node tersebut.

### 3. Random Forest Regressor
Random forest regressor adalah **Model ensemble** yang menggabungkan prediksi dari banyak *decision tree* (biasa disebut *base estimators*) dan menggunakan teknik *bagging* dan *random feature selection* untuk meningkatkan stabilitas dan akurasi.  

#### Cara Kerja
1. **Bagging (*Bootstrap Aggregating*)**:  
   - Setiap pohon dilatih pada subset data acak (*bootstrapping*).  
2. **Random Feature Selection**:  
   - Saat pemecahan node, hanya subset acak fitur (`max_features`) yang dipertimbangkan.  
3. **Aggregasi**:  
   - Prediksi akhir adalah rata-rata prediksi semua pohon.  

Perhatikan code di bawah ini:
```
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}
```
Pada algoritma Decision Tree dan Random Forest, parameter random_state ditetapkan untuk menjamin reproduksibilitas hasil eksperimen. Parameter lainnya pada kedua algoritma tersebut dipertahankan pada nilai default. Untuk Linear Regression, seluruh parameter menggunakan nilai default tanpa penyesuaian khusus.

```
results = {}
```
Selanjutnya, menyiapkan sebuah dictionary untuk menyimpan hasil evaluasi.

Setelah itu, melatih dan mengevaluasi ketiga algoritma machine learning, lalu menyimpan hasil evaluasinya ke dalam dictionary yang telah disiapkan sebelumnya.

### Hyperparameter Tuning
Algoritma yang dipilih untuk melalui tahap tuning adalah Random Forest. Metode hyperparameter tuning yang akan digunakan adalah Grid Search. Grid Search merupakan metode hyperparameter tuning yang mencoba semua kombinasi hyperparameter yang telah ditentukan, kemudian mengevaluasi performa model untuk setiap kombinasi tersebut. Hasil akhirnya akan memilih kombinasi hyperparameter terbaik.

Evaluasi model pada tahapan gridsearch menggunakan cross-validation agar model robust. Proses hyperparameter tuning menggunakan Grid Search menghasilkan kombinasi hyperparameter terbaik untuk model RandomForestRegressor sebagai berikut:
```
RandomForestRegressor(max_depth=15, max_features=0.5, min_samples_split=5,
                      n_estimators=300, random_state=42)
```
Kombinasi optimal tersebut terdiri dari parameter:
* max_depth = 15
* max features=0.5
* in_samples_split=5
* n_estimators=300
* random_state=42

## **Evaluation**

### **Metrik Evaluasi yang Digunakan**
Dalam evaluasi model regresi yang digunakan, dua metrik utama digunakan untuk menilai performa model:

#### **1. Mean Absolute Error (MAE)**
MAE mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual. Metrik ini memberikan gambaran seberapa jauh nilai prediksi dari nilai sebenarnya tanpa mempertimbangkan arah kesalahan.

Rumus MAE:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
$$

Di mana:
- $n$ adalah jumlah sampel  
- $y_i$ adalah nilai aktual  
- $\hat{y_i}$ adalah nilai prediksi  

Semakin kecil nilai MAE, semakin baik model dalam melakukan prediksi.

#### **2. Mean Squared Error (MSE)**
MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. MSE memberikan penalti lebih besar pada kesalahan yang besar, sehingga lebih sensitif terhadap outlier.

Rumus MSE:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

Di mana:
- $n$ adalah jumlah sampel  
- $y_i$ adalah nilai aktual  
- $\hat{y_i}$ adalah nilai prediksi  

Semakin kecil nilai MSE, semakin baik performa model dalam melakukan prediksi.

### **Hasil Evaluasi Model Melalui Proses Hyperparameter Tuning pada Algoritma Random Forest**
```
=== Random Forrest ===
Train MAE: 0.5625
Train MSE: 1.7246
```

Jika kita perhatikan kembali, hasil model yang diperoleh dari proses hyperparameter tuning menunjukkan peningkatan nilai MAE dan MSE dibandingkan sebelum dilakukan tuning. Berikut perbandingannya:

Sebelum Tuning
```
=== Random Forest ===

Train MAE: 0.4497

Train MSE: 0.8033
```
Setelah Tuning
```
=== Random Forrest ===

Train MAE: 0.5625

Train MSE: 1.7246
```
Hal ini umum terjadi dalam proses hyperparameter tuning. Evaluasi hasil hyperparameter tuning menunjukkan tidak adanya peningkatan performa model; sebaliknya, proses tersebut justru menyebabkan penurunan kinerja prediktif. Oleh karena itu, kombinasi hiperparameter hasil tuning tidak diimplementasikan. Konfigurasi hiperparameter default dipilih kembali untuk mempertahankan stabilitas model sesuai dengan kinerja awal yang telah teruji.

### **Hasil Evaluasi Model**
Tiga model telah diuji dengan menggunakan data training dan testing:

| Model                     | MAE (Train) | MAE (Test) | MSE (Train) | MSE (Test) |
|--------------------------|------------|------------|------------|------------|
| Linear Regression       | 1.0910     | 1.0869     | 5.5161     | 5.2402     |
| Decision Tree Regressor | 0.0000     | 1.8519     | 0.0000     | 13.8315    |
| Random Forest Regressor | 0.4497     | 1.1835     | 0.8033     | 5.7330     |

### **Analisis Hasil dan Pemilihan Model Terbaik**
Dari tabel di atas, dapat diamati bahwa:
- **Linear Regression** menunjukkan performa yang stabil antara data training dan testing, dengan perbedaan error yang kecil. Hal ini menunjukkan bahwa model ini *good fit* dan memiliki generalisasi yang baik.
- **Decision Tree Regressor** menunjukkan MAE dan MSE yang sangat rendah pada data training, tetapi nilai errornya meningkat tajam pada data testing, yang mengindikasikan **overfitting**.
- **Random Forest Regressor** lebih baik dibandingkan Decision Tree karena memiliki performa yang lebih stabil di antara data training dan testing, tetapi masih memiliki error yang lebih tinggi dibandingkan Linear Regression.

Dari hasil ini, **Linear Regression dipilih sebagai model terbaik** karena memiliki performa yang konsisten dan generalisasi yang lebih baik dibandingkan model lainnya.

### **Hubungan dengan Business Understanding**
#### **Apakah Model Menjawab Problem Statement?**
Ya, model yang dikembangkan berhasil memprediksi nilai ujian akhir mahasiswa berdasarkan berbagai faktor akademik seperti kehadiran, nilai tugas, dan partisipasi kelas. Dengan akurasi yang baik, model ini dapat membantu universitas dalam mengidentifikasi mahasiswa dengan risiko akademik tinggi lebih awal.

#### **Apakah Model Mencapai Goals yang Ditetapkan?**
- **Prediksi nilai ujian akhir mahasiswa**: Model berhasil memprediksi dengan error yang cukup rendah, terutama dengan Linear Regression.
- **Identifikasi faktor dominan**: Berdasarkan analisis regresi, fitur seperti *Attendance* dan *Previous_Scores* memiliki korelasi yang tinggi terhadap hasil akhir mahasiswa.
- **Rekomendasi berbasis data**: Dengan hasil ini, universitas dapat membuat kebijakan intervensi lebih awal terhadap mahasiswa dengan performa rendah.

#### **Dampak Model terhadap Universitas**
Dengan implementasi model ini, universitas dapat:
1. **Mengurangi angka mahasiswa gagal dan Drop Out (DO)** dengan memberikan dukungan akademik lebih dini.
2. **Meningkatkan efisiensi evaluasi akademik** dengan melakukan pemantauan berbasis data secara real-time.
3. **Membantu mahasiswa meningkatkan performa akademik** dengan rekomendasi yang tepat sesuai dengan prediksi model.

### **Kesimpulan**
Dari hasil evaluasi model, **Linear Regression merupakan model terbaik** dalam prediksi nilai ujian akhir mahasiswa karena memiliki performa yang stabil dan error yang rendah. Model ini dapat diterapkan dalam sistem akademik universitas untuk memantau dan mengidentifikasi mahasiswa yang berisiko akademik tinggi, sehingga memungkinkan intervensi lebih awal untuk meningkatkan tingkat kelulusan mereka.
