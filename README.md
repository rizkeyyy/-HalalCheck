# 🍽️ HalalCheck – Cek Kehalalan Produk Secara Otomatis
HalalCheck adalah sistem berbasis Deep Learning yang memungkinkan pengguna untuk mengecek status halal atau haram suatu produk makanan melalui analisis daftar bahan yang diperoleh dari label kemasan. Menggunakan kombinasi OCR (Google Vision API) untuk ekstraksi teks, LSTM dengan Multi-Head Attention untuk klasifikasi berbasis NLP, serta Streamlit sebagai antarmuka pengguna.

## Latar Belakang Masalah
Seiring dengan meningkatnya kesadaran masyarakat Muslim terhadap kehalalan makanan, masih banyak produk di pasaran yang tidak memiliki label halal resmi atau memiliki komposisi bahan yang meragukan. <br>
Banyak konsumen kesulitan dalam memeriksa status halal/haram produk secara mandiri, terutama jika bahan makanan tertulis dalam istilah teknis atau kode E.<br>
Oleh karena itu, proyek ini dikembangkan untuk memberikan solusi cepat dan otomatis dalam mengecek status halal/haram suatu produk makanan berdasarkan daftar komposisi yang tertera pada kemasan.

## Tujuan Proyek
✅ Mengembangkan sistem berbasis OCR dan NLP untuk mengidentifikasi status halal atau haram suatu makanan. <br>
✅ Menggunakan Google Cloud Vision API untuk mengekstrak teks dari label makanan secara otomatis. <br>
✅ Menggunakan model LSTM + Multi-Head Attention agar sistem dapat menganalisis daftar bahan dan menentukan kehalalan produk. <br>
✅ Memberikan peringatan otomatis jika ditemukan bahan yang meragukan dalam daftar komposisi makanan. <br>
✅ Menyimpan hasil analisis ke dalam database PostgreSQL untuk referensi lebih lanjut. <br>
✅ Menyediakan antarmuka interaktif yang memudahkan pengguna dalam memasukkan teks manual atau mengunggah gambar untuk dianalisis.

## Dataset
Dataset yang digunakan dalam proyek ini mencakup daftar bahan makanan halal, haram, dan meragukan, yang dikumpulkan dari berbagai sumber terpercaya. <br>
Dataset ini telah diproses agar sesuai untuk pelatihan model NLP berbasis LSTM, dengan tokenizer dan word embedding untuk meningkatkan pemahaman model terhadap daftar bahan makanan.

## Teknologi yang Digunakan
🔹 Python → Bahasa pemrograman utama. <br>
🔹 TensorFlow & Keras → Membangun model **Deep Learning berbasis LSTM + Multi-Head Attention**. <br>
🔹 Google Cloud Vision API → Ekstraksi teks dari label makanan menggunakan **OCR otomatis**. <br>
🔹 Natural Language Processing (NLP) → Preprocessing teks untuk **membersihkan data dan menganalisis komposisi bahan**. <br>
🔹 PostgreSQL → Penyimpanan hasil analisis agar dapat diakses kembali. <br>
🔹 Streamlit → Deploy aplikasi berbasis web di **Hugging Face Spaces**. <br>
🔹 Matplotlib & Seaborn → Visualisasi data dan analisis performa model. <br>

## Evaluasi Model
Model LSTM yang digunakan telah diuji dan menunjukkan hasil yang **sangat baik** dalam mengklasifikasikan status **halal atau haram** dari daftar bahan makanan.

- Training Accuracy: 98.36%
- Validation Accuracy: 98.57%
- Test Accuracy: 99.10%

### Analisis Confusion Matrix:
- Kesalahan klasifikasi sangat rendah, menandakan model dapat membedakan kelas dengan baik.
- Salah klasifikasi hanya terjadi pada 17 sampel Halal dan 38 sampel Haram dari 3.979 data uji.

### Classification Report

| **Kelas**  | **Precision** | **Recall** | **F1-Score** | **Support** |
|------------|--------------|------------|--------------|------------|
| **Halal**  | 99%          | 99%        | 99%          | 1,796      |
| **Haram**  | 99%          | 99%        | 99%          | 2,183      |


### Kesimpulan:
- Precision tinggi → Model jarang salah dalam memprediksi produk Halal/Haram.
- Recall tinggi → Model dapat mengenali hampir semua produk dengan benar.
- Akurasi model sangat baik, siap digunakan untuk klasifikasi halal/haram makanan.

## Fitur Utama
✔ Ekstraksi teks otomatis dari label makanan menggunakan Google Vision API (OCR). <br>
✔ Klasifikasi bahan makanan berdasarkan model LSTM + Multi-Head Attention. <br>
✔ Peringatan otomatis jika ditemukan bahan yang meragukan atau haram. <br>
✔ Mendukung input manual & upload gambar untuk fleksibilitas pengguna. <br>
✔ Penyimpanan hasil analisis ke database PostgreSQL untuk referensi lebih lanjut. <br>
✔ Antarmuka interaktif berbasis Streamlit, di-deploy di Hugging Face Spaces.

## Cara Menggunakan Aplikasi
![Demo GIF](Demo.gif)

Coba Aplikasi di <a href="https://huggingface.co/spaces/rizkystiawanp/HalalCheck">Huggingface</a> <br>

1️⃣ Upload gambar label makanan atau masukkan daftar bahan secara manual. <br>
2️⃣ OCR akan mengekstrak teks dari label produk (jika menggunakan gambar). <br>
3️⃣ Model NLP berbasis LSTM + Multi-Head Attention akan menganalisis daftar bahan. <br>
4️⃣ Sistem memberikan hasil klasifikasi dan peringatan otomatis jika ada bahan meragukan. <br>
5️⃣ Hasil analisis akan disimpan ke database PostgreSQL untuk referensi lebih lanjut.

## Pengembangan di Masa Depan
🔹 Integrasi dengan database halal resmi untuk meningkatkan keakuratan analisis. <br>
🔹 Pengembangan aplikasi mobile agar lebih mudah digunakan di mana saja. <br>
🔹 Fitur barcode scanner untuk mempermudah pengecekan kehalalan produk secara instan. <br>
🔹 Peningkatan dataset dan model NLP untuk meningkatkan akurasi klasifikasi. <br>
🔹 Menambahkan fitur multi-bahasa agar dapat digunakan secara global. <br>
🔹 Menyimpan hasil analisis lebih lengkap di database agar bisa diakses kembali oleh pengguna.

## ⚠ Disclaimer
🔸 Aplikasi ini bukan pengganti sertifikasi halal resmi dan hanya bertujuan untuk memberikan informasi awal kepada pengguna. <br>
🔸 Hasil klasifikasi tidak bersifat mutlak, karena regulasi halal di berbagai negara bisa berbeda. <br>
🔸 Pengguna disarankan untuk tetap memeriksa sertifikasi halal resmi dari otoritas terkait sebelum mengonsumsi produk.

## Dikembangkan Oleh
**[Rizky SP](https://github.com/rizkystiawanp)** : Data Scientist <br>
**[Dzaki A.](https://github.com/DzakiAF)** : Data Scientist <br>
**[Dwi H.](https://github.com/dwihst)** : Data Analyst <br>
**[Aditya W.](https://github.com/bUtekwijay)** : Data Engineer <br>
**[Rizki R.](https://github.com/rizkeyyy)** : Data Engineer

[Lihat presentasi sebagai PDF](presentation.pdf)