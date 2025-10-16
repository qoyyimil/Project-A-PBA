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

**5. Data Splitting**

**6. Data Balancing**

**7. Implementation BERT**

**8. Evaluation**

## ✅ Hasil dan Pembahasan
**1. Data Scraping**

**2. Preprocessing**

**3. EDA (Exploratory Data Analysis)**

**4. Data Labelling**

**5. Data Splitting**

**6. Data Balancing**

**7. Implementation BERT**

**8. Evaluation**

## 📈 Kesimpulan dan Saran
**1. Kesimpulan**

**2. Saran**
