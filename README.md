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

**a. EDA Awal (Audit Kualitas Data)**
- Tujuan: Mengidentifikasi anomali, missing value, dan noise (kata-kata tidak relevan) pada teks mentah.
- Metode: Menghapus data kosong/duplikat, menghitung dan memvisualisasikan frekuensi kata (Word Cloud dan Bar Chart), serta menganalisis distribusi panjang teks mentah. Hasilnya membenarkan perlunya Stopword Removal karena dominasi kata umum.

**b. Validasi Pasca-Preprocessing**
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

**a. Model Utama:** DistilBertForSequenceClassification (distilbert-base-uncased).

**b. Persiapan Data Khusus BERT:**
- Tokenisasi: Teks dipecah dan dikonversi menjadi ID numerik menggunakan DistilBertTokenizer.
- Normalisasi Input: Panjang sequence diseragamkan (max_length=128) menggunakan Padding dan Truncation.
- Batching: Data diubah menjadi PyTorch Tensors dan disajikan per batch (ukuran 16) melalui DataLoader.

**c. Proses Pelatihan (Fine-Tuning):**
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
- Hasil dari scraping artikel 
<img width="1622" height="616" alt="image" src="https://github.com/user-attachments/assets/de59476c-6034-4fec-8585-0fc5dedddbdc" />

**2. EDA (Exploratory Data Analysis) Hasil Pre-processing**

**a. EDA Tahap Awal dan Audit Kualitas Data**
<img width="1465" height="583" alt="image" src="https://github.com/user-attachments/assets/823843d3-833d-4a61-85cb-1d0d20204760" />
- Visualisasi diatas menunjukkan dominasi Stopwords ("the", "and") merupakan frekuensi besar stopword yang tidak memiliki makna kontekstual terhadap analisis sentimen.
  
<img width="1392" height="416" alt="image" src="https://github.com/user-attachments/assets/35ed2b83-bd24-400f-ab75-dd16df47470a" />
- Analisis menunjukkan variasi ekstrem dalam panjang artikel (min 55 kata, maks 3.133 kata, rata-rata 819 kata), yang memerlukan Padding dan Truncation yang tepat pada model BERT.


**b. Validasi Pra-pemrosesan dan Analisis Fitur Teks**
<img width="1071" height="418" alt="image" src="https://github.com/user-attachments/assets/c8d42d18-117d-4811-9fb7-6e6d821b45c5" />
- Visualisasi di atas menunjukkan kata-kata umum (noise) seperti "the" dan "and" tidak lagi muncul.
- Kata-kata yang paling dominan saat ini adalah yang relevan dengan topik, yaitu "uber" (2.378 kali) dan "driver" (1.277 kali). Hal ini mengonfirmasi data yang dihasilkan bersih dan kontekstual.

<img width="1424" height="407" alt="image" src="https://github.com/user-attachments/assets/8fe72815-b17b-4bee-9b53-a203021209f1" />
- Terjadi penurunan signifikan dalam panjang rata-rata teks (dari 819 kata menjadi 408 kata), dengan distribusi yang lebih seragam. Hal ini membuat teks lebih efisien dan konsisten untuk diproses oleh model DistilBERT.



**3. Data Labelling**

**a. Distribusi Label Sentimen**
<img width="857" height="387" alt="image" src="https://github.com/user-attachments/assets/45e2035f-74d7-435c-aaef-5dcf54f5b23b" />
| Label Sentimen | Jumlah Artikel (Count) | Persentase Proporsi |
| :--- | :---: | :---: |
| **POSITIF** | 109 | 63.74% |
| **NEGATIF** | 62 | 36.26% |
| **Total** | **171** | **100.00%** |

**b. Hasil Analisis Sentimen setelah Labelling**
<img width="1350" height="413" alt="image" src="https://github.com/user-attachments/assets/d022a36e-4148-4698-ad7f-fb764e657355" />
- Konsentrasi Topik: Pemberitaan sangat terfokus pada topik Tech (107 artikel/62.6%) dan Business (24 artikel/14.0%).
- Sentimen Dominan: Meskipun POSITIF adalah sentimen mayoritas di kedua topik utama (misalnya, 69 Positif pada topik Tech), terdapat volume NEGATIF yang signifikan (38 Negatif pada topik Tech).

**4. Data Splitting**
| Set Data | Jumlah Data | Proporsi | Sentimen Positif (%) | Sentimen Negatif (%) |
| :--- | :---: | :---: | :---: | :---: |
| Training | 119 | 70% | 63.87 | 36.13 |
| Validation | 26 | 15% | 65.38 | 34.62 |
| Test | 26 | 15% | 61.54 | 38.46 |

**5. Data Balancing**
| Label Sentimen | Jumlah Sebelum Augmentasi | Jumlah Sesudah Augmentasi |
| :--- | :---: | :---: |
| Positif | 76 | 76 |
| Negatif | 43 | 76 |
| **Total** | **119** | **152** |

**6. Implementation BERT**
- Model & Data Input: Model DistilBertForSequenceClassification (distilbert-base-uncased) berhasil dimuat dan menerima data latih yang sudah di-tokenize dan di-batch (batch size 16) sesuai dengan format PyTorch Tensors.
- Konfirmasi Pelatihan: Proses Fine-Tuning berhasil dieksekusi selama 4 Epoch dengan Optimizer AdamW (Learning Rate $2e-5$).
- Output: Pelatihan menghasilkan weights model yang telah diperbarui dan siap diukur kemampuan generalisasinya pada tahap evaluasi berikutnya.

**7. Evaluation**

**a. Hasil Pelatihan dan Validasi**
| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.6831 | 60.53% | 0.6647 | 73.08% |
| 2 | 0.6288 | 80.92% | 0.5885 | 80.77% |
| 3 | 0.5517 | 76.32% | 0.5198 | 80.77% |
| 4 | 0.5062 | 80.26% | 0.4902 | 80.77% |

**b. Hasil Pengujian pada Data Uji**
| Kelas/Metrik | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **NEGATIF** | 0.70 | 0.70 | 0.70 | 10 |
| **POSITIF** | 0.81 | 0.81 | 0.81 | 16 |
| **Accuracy** | | | 0.77 | 26 |
| **Macro Avg** | 0.76 | 0.76 | 0.76 | 26 |
| **Weighted Avg** | 0.77 | 0.77 | 0.77 | 26 |
- Model mencapai akurasi keseluruhan sebesar 77% (20 dari 26 artikel diklasifikasikan dengan benar).

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/486e5f7d-9901-42ae-8460-955db5c4a9d1" />
- Model menunjukkan kinerja yang stabil dan tidak bias berkat Data Balancing:
F1-Score Positif: 0.81 dan F1-Score Negatif: 0.70
- Jumlah kesalahan False Positive (3 kasus) dan False Negative (3 kasus) adalah seimbang, yang mengonfirmasi bahwa model tidak memiliki kecenderungan bias terhadap salah satu kelas sentimen.

**c. Analisis Kesalahan (Error Analysis)**

Analisis kualitatif terhadap kasus kesalahan prediksi model mengungkapkan dua pola kelemahan utama:
### Tabel Analisis Kesalahan (*Error Analysis*)

| Pola Kelemahan Model | Deskripsi Singkat | Tipe Kesalahan yang Dominan |
| :--- | :--- | :---: |
| **Kesulitan Masalah vs. Solusi** | Model terpaku pada kata positif (e.g., "safety") saat artikel membahas solusi untuk masalah negatif. | False Positive (FN) |
| **Ketergantungan Konflik** | Model bias terhadap istilah konflik (e.g., "lawsuit", "labor") sehingga salah mengklasifikasikan hasil positif menjadi NEGATIF. | False Negative (FP) |

## 📈 Kesimpulan dan Saran
**1. Kesimpulan**
- Model DistilBERT terbukti efektif untuk tugas klasifikasi sentimen pada domain berita, mencapai Akurasi keseluruhan 77% pada data uji.
- Teknik Synonym Replacement berperan krusial dalam mengatasi class imbalance. Hasilnya, model menunjukkan kinerja yang seimbang (F1-Score 0.81 untuk Positif dan 0.70 untuk Negatif).
- Analisis kesalahan mengonfirmasi keterbatasan model dalam memahami konteks sentimen yang ambigu/campuran (positif muncul sebagai respons terhadap masalah negatif).

**2. Insight**
- Berita korporat seringkali tidak monolitik (Positif/Negatif). Insight dari Error Analysis menunjukkan perlunya analisis yang melampaui level dokumen untuk mendapatkan pemahaman yang benar-benar berguna bagi bisnis.
- Keberhasilan model ringan (DistilBERT) membuktikan bahwa untuk aplikasi bisnis, model yang lebih efisien sudah mampu memberikan gambaran sentimen yang andal dengan biaya komputasi yang jauh lebih rendah daripada model skala besar.
- Penggunaan VADER sebagai alat pelabelan otomatis awal adalah strategi yang sangat pragmatis dan efektif untuk proyek dengan sumber daya terbatas, menghasilkan ground truth yang memadai untuk melatih model deep learning yang canggih.

**3. Saran**
- Disarankan beralih ke Aspect-Based Sentiment Analysis (ABSA) untuk mengatasi sentimen campuran, yaitu mengidentifikasi sentimen terhadap aspek spesifik (misalnya, sentimen NEGATIF terhadap 'keamanan' dalam satu artikel).
- Membuat gold-standard dataset kecil dengan anotasi manual (human annotators) untuk memvalidasi akurasi VADER dan meningkatkan keandalan ground truth.
- Bereksperimen dengan model Transformer yang lebih besar (misalnya, RoBERTa) yang berpotensi memiliki pemahaman nuansa bahasa yang lebih baik dan dapat meningkatkan performa pada kasus-kasus sulit.

## 🔧 Tools dan Teknologi
- **Bahasa Pemrograman:** Python
- **Model Utama:** DistilBERT-base-uncased (*Fine-Tuning*)
- **Library Kunci:** `HuggingFace Transformers`, `PyTorch`, `Scikit-learn`, `NLTK` (VADER & Lemmatization), `BeautifulSoup4`, `Pandas`
- **Metode Kunci:** Web Scraping, Data Augmentation (Synonym Replacement)


