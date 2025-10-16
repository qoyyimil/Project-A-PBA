# 🚀 README : ANALISIS SENTIMEN ARTIKEL BERITA UBER

## 🎯 Judul 
**Analisis Sentimen Artikel Berita Mengenai Uber Menggunakan Fine-Tuning Model DistilBERT dengan Teknik Data Augmentation**

## 🧑‍💻 Anggota Kelompok 6 - PBA
| Nama Lengkap | NIM |
| :--- | :--- |
| **Sintiarani Febyan Putri** | `5026221044` |
| **Qoyyimil Jamilah** | `5026221115` |

## 📝 Abstrak
Proyek ini bertujuan mengklasifikasikan sentimen berita mengenai Uber menggunakan deep learning. Data 171 artikel berita dikumpulkan, diproses ekstensif, dan dilabeli otomatis oleh VADER. Untuk mengatasi ketidakseimbangan data (63.74% Positif), diterapkan teknik Synonym Replacement pada data latih. Model utama, DistilBERT-base-uncased, di-fine-tune pada data yang seimbang. Hasil evaluasi model mencapai Akurasi 77%, dengan F1-Score 0.81 (Positif) dan 0.70 (Negatif), membuktikan efektivitas data augmentation dalam menghasilkan model klasifikasi yang stabil.

## ⚙️ Metode Penelitian
![User melakukan registrasi (1)](https://github.com/user-attachments/assets/ad251300-b389-4426-9846-519c808a0f3c)


**1. Data Scraping**

Tahap ini berfokus pada pengumpulan dataset teks dari sumber berita internasional terpercaya.
- Target Data: 171 artikel berita berbahasa Inggris mengenai Uber.
- Sumber Data: Tiga portal berita internasional kredibel yaitu CNN, CNBC, dan NBC News.
- Strategi Pencarian: Menggunakan Operator Pencarian Google (site:www.cnn.com "Uber") untuk memastikan artikel yang dikumpulkan relevan dan spesifik dari masing-masing portal berita.
- Metode: Dilakukan menggunakan teknik Web Scraping dengan bahasa pemrograman Python (memanfaatkan libraries seperti requests dan beautifulsoup4).

**2. Preprocessing**

Tahap ini berfungsi merapikan dan menstandarisasi dataset teks agar efisien diproses oleh model DistilBERT.
- Expansion of Contractions: Mengubah singkatan kata dalam Bahasa Inggris (misalnya: "don't" menjadi "do not") untuk memastikan konsistensi makna kalimat.
- Tokenization & Cleaning: Memecah teks menjadi potongan kata (tokens) sekaligus menghapus elemen yang tidak relevan (URL, angka, tanda baca) dan menyeragamkan huruf menjadi huruf kecil (lowercasing).
- Stopword Removal & Lemmatization: Menghapus kata umum yang tidak signifikan (stopwords) tetapi mempertahankan kata negasi (misalnya: "not") agar sentimen tidak hilang. Proses diikuti dengan Lemmatization (mengembalikan kata ke bentuk dasar).
- Final Output: Teks disusun kembali (Text Reconstruction) dan disimpan dalam kolom Teks_Final_Clean yang siap untuk pelabelan sentimen.

**3. EDA (Exploratory Data Analysis)**

EDA dilakukan untuk memahami struktur data dan mengaudit kualitasnya, dibagi menjadi dua fase penting:

a. EDA Awal (Audit Kualitas Data) 
- Tujuan: Mengidentifikasi anomali, missing value, dan noise (kata-kata tidak relevan) pada teks mentah.
- Metode: Menghapus data kosong/duplikat, menghitung dan memvisualisasikan frekuensi kata (Word Cloud dan Bar Chart), serta menganalisis distribusi panjang teks mentah. Hasilnya membenarkan perlunya Stopword Removal karena dominasi kata umum.

b. Validasi Pasca-Preprocessing
- Tujuan: Memastikan efektivitas proses pembersihan teks (Teks_Final_Clean) dan bahwa noise telah berhasil dihilangkan.
- Metode: Menganalisis kembali frekuensi kata (Word Cloud dan Bar Chart) pada data bersih untuk memastikan "Uber" dan kata kunci relevan lainnya mendominasi. Analisis ini juga memverifikasi konsistensi panjang teks setelah dibersihkan.

**4. Data Labelling**

Tahap ini memberikan label sentimen (ground truth) pada data yang bersih secara cepat dan objektif.
- Metode Utama: Menggunakan algoritma berbasis leksikon VADER (Valence Aware Dictionary and sEntiment Reasoner) dari pustaka NLTK.
- Mekanisme: VADER menganalisis teks untuk menghasilkan compound score yang mencerminkan polaritas emosi keseluruhan.
- Aturan Klasifikasi:
Positif dengan Compound Score $\ge 0.05$;
Negatif dengan Compound Score $\le -0.05$;
Teks netral di antara ambang batas tidak diikutsertakan dalam pemodelan biner

**5. Data Splitting**

Rasio Pembagian:
- Data Latih (Training): 70%
- Data Validasi (Validation): 15%
- Data Uji (Test): 15%

**6. Data Balancing**

Tahap ini bertujuan mencegah bias model yang timbul dari ketidakseimbangan jumlah sampel antar kelas (Positif vs. Negatif).
- Masalah: Data Latih mengalami class imbalance (Positif mayoritas, Negatif minoritas).
- Metode: Menerapkan teknik Augmentasi Data dengan Penggantian Sinonim (Synonym Replacement).
- Prosedur Kunci: Augmentasi dilakukan secara eksklusif pada Data Latih kelas minoritas hingga mencapai rasio 1:1.
- Tujuan: Menghasilkan data latih yang seimbang secara artifisial, memungkinkan model DistilBERT mempelajari karakteristik kedua kelas sentimen secara setara.

**7. Implementation BERT**

Tahap ini merupakan inti pemodelan menggunakan Transfer Learning dari arsitektur Transformer.

a. Model Utama: DistilBertForSequenceClassification (distilbert-base-uncased).

b. Persiapan Data Khusus BERT:
- Tokenisasi: Teks dipecah dan dikonversi menjadi ID numerik menggunakan DistilBertTokenizer.
- Normalisasi Input: Panjang sequence diseragamkan (max_length=128) menggunakan Padding dan Truncation.
- Batching: Data diubah menjadi PyTorch Tensors dan disajikan per batch (ukuran 16) melalui DataLoader.

c. Proses Pelatihan (Fine-Tuning):
- Tujuan: Memperbarui weights model pra-terlatih agar spesifik mengenali sentimen berita Uber.
- Hyperparameter Kunci: Dilatih selama 4 Epoch, menggunakan Optimizer AdamW dengan Learning Rate $2e-5$.
- Kontrol Kualitas: Evaluasi dilakukan pada data validasi di setiap epoch untuk memantau loss dan mencegah overfitting.

**8. Evaluation**

Basis evaluasi dilakukan pada data uji (Test Set) yang merupakan data yang belum pernah dilihat model untuk menilai kemampuan generalisasi model. Matrix kunci nya yaitu: 
- Accuracy: Persentase prediksi yang benar secara keseluruhan.
- Precision & Recall: Mengukur ketepatan dan sensitivitas model pada setiap kelas.
- F1-Score: Rata-rata harmonik Precision dan Recall, krusial untuk data dengan kelas yang tidak seimbang.
- Confusion Matrix: Memvisualisasikan jenis kesalahan yang dibuat model (TP, TN, FP, FN).

## ✅ Hasil dan Pembahasan
**1. Data Scraping**
<img width="1622" height="616" alt="image" src="https://github.com/user-attachments/assets/de59476c-6034-4fec-8585-0fc5dedddbdc" />

**2. EDA (Exploratory Data Analysis) Hasil Pre-processing**

a. EDA Tahap Awal dan Audit Kualitas Data
<img width="1465" height="583" alt="image" src="https://github.com/user-attachments/assets/823843d3-833d-4a61-85cb-1d0d20204760" />
<img width="841" height="470" alt="image" src="https://github.com/user-attachments/assets/f4e1feec-85d9-4845-ba85-2e349449f883" />

b. Validasi Pra-pemrosesan dan Analisis Fitur Teks
<img width="1071" height="418" alt="image" src="https://github.com/user-attachments/assets/c8d42d18-117d-4811-9fb7-6e6d821b45c5" />
<img width="859" height="476" alt="image" src="https://github.com/user-attachments/assets/40174fb4-8286-4c56-b9d4-753cb989e21b" />

**4. Data Labelling**

a. Distribusi Label Sentimen
<img width="694" height="660" alt="image" src="https://github.com/user-attachments/assets/40ce3003-f9f5-410d-a957-243fbd578713" />

b. Hasil Analisis Sentimen setelah Labelling
<img width="989" height="590" alt="download (9)" src="https://github.com/user-attachments/assets/6733eae9-7b85-44b1-92bf-21b581189937" />


**5. Data Splitting**

**6. Data Balancing**

**7. Implementation BERT**

**8. Evaluation**

## 📈 Kesimpulan dan Saran
**1. Kesimpulan**

**2. Saran**
