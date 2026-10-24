# 🏫 Sistem Presensi Biometrik Wajah Siswa SMK PGRI Sukoharjo
### Komparasi Multi-Model: ArcFace, InsightFace (Native ONNX), dan FaceNet

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/Engine-InsightFace%20ONNX-green.svg)](https://github.com/deepinsight/insightface)
[![DeepFace](https://img.shields.io/badge/Framework-DeepFace-orange.svg)](https://github.com/serengil/deepface)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)

Purwarupa (*prototype*) aplikasi web presensi biometrik pengenalan wajah nirsentuh (*touchless*) yang dikembangkan sebagai bagian dari penelitian Tugas Akhir Skripsi di **Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Kristen Duta Wacana (UKDW)**.

Aplikasi ini mengimplementasikan metode **Transfer Learning** dan komparasi serentak (*simultaneous head-to-head evaluation*) terhadap tiga arsitektur *deep learning* pengenal wajah terkemuka: **ArcFace**, **InsightFace (Native ONNX)**, dan **FaceNet** dalam menghadapi variasi visual nyata siswa di lingkungan sekolah (kondisi normal, penggunaan hijab, kacamata, masker penutup wajah, dan pencahayaan kelas).

---

## 📌 Fitur Utama & Inovasi Sistem

- 📷 **Pindai Presensi Multi-Model Serentak (Head-to-Head):** Satu kali pengambilan foto langsung dievaluasi oleh 3 arsitektur model secara serentak untuk membandingkan skor jarak, status verifikasi, dan kecepatan latensi komputasi.
- 🎯 **Kunci Identitas Tunggal (*Single Identity Consensus Lock*):** Mencegah *bug* foto terbaca menjadi dua nama siswa berbeda. Sistem mengunci satu kandidat identitas utama (berbasis ArcFace/konsensus mayoritas) dan mengevaluasi kesepakatan model lain terhadap identitas tersebut.
- 😷 **Deteksi Masker Nirsentuh (*Autonomous Mask Recognition*):** Menggunakan analisis kromatik ruang warna kulit $YCrCb$ untuk mendeteksi penutup wajah secara otomatis tanpa perlu mencentang *checkbox*. Sistem langsung mengalihkan komputasi ke **Jalur Fitur Periokular** (fokus area mata & dahi).
- 🪞 **Pratinjau & Tangkapan Kamera Cermin (*Mirror Camera Feed*):** Injeksi CSS dan pemrosesan `ImageOps.mirror` memastikan *live preview* dan hasil jepretan kamera berorientasi cermin (*flip horizontal*) untuk pengalaman pengguna yang intuitif.
- ⚡ **Integrasi Mesin Cepat InsightFace (Native ONNX):** Menggantikan arsitektur lama VGG-Face dengan model resmi InsightFace (*Buffalo_L*). Latensi pemindaian terpangkas drastis dari $\approx 890\text{ ms}$ menjadi hanya **$\pm 85\text{ ms}$**.
- ➕ **Registrasi Sekali untuk Semua Model (Dual-Template):** Satu foto pendaftaran otomatis mengekstrak representasi vektor Wajah Penuh dan Wilayah Periokular untuk ketiga model sekaligus ke dalam basis data SQLite.
- 📊 **Laporan Empiris & Rekomendasi Model (Modul 5):** Menyediakan visualisasi metrik akurasi riil, tabel log evaluasi konsensus siswa, dan tombol ekspor laporan `.CSV` untuk lampiran Bab 4 Skripsi.
- ⚙️ **Kalibrasi Ambang Batas Terpusat (Modul 6):** Panel pengaturan nilai ambang batas (*threshold tuning*) dinamis berbasis `st.session_state` dengan opsi *reset to default*.

---

## 🛠️ Arsitektur & Spesifikasi Model AI

| Parameter | ArcFace (Keras) | InsightFace (Native ONNX) ⚡ | FaceNet (Inception) |
| :--- | :--- | :--- | :--- |
| **Arsitektur Dasar** | ResNet-50 | ResNet-50 (*Buffalo_L*) | Inception-ResNet-v1 |
| **Fungsi Loss** | *Additive Angular Margin* | *Additive Angular Margin* | *Triplet Loss* |
| **Detektor Wajah** | MTCNN (*5 Facial Landmarks*) | SCRFD (*High-speed Detection*) | MTCNN (*5 Facial Landmarks*) |
| **Dimensi Embedding**| 512 Dimensi | 512 Dimensi (Ternormalisasi) | 128 / 512 Dimensi |
| **Metrik Jarak** | *Cosine Distance* | *Cosine Distance* | *Euclidean Distance* |
| **Ambang Standar** | `0.4800 – 0.5000` | `0.4500 – 0.5000` | `9.5000 – 10.0000` |
| **Rata-rata Latensi** | $\pm 399\text{ ms}$ | **$\pm 88\text{ ms}$ (Tercepat)** | $\pm 302\text{ ms}$ |

---

## 📂 Struktur Direktori Proyek

```text
Face_Recog-main/
├── app.py                   # Berkas utama antarmuka Streamlit (6 Modul)
├── database.py              # Modul SQLite (tabel siswa & log kehadiran)
├── requirements.txt         # Daftar dependensi Python
├── README.md                # Dokumentasi komprehensif sistem
└── presensi_smk.db          # Basis data SQLite lokal (dihasilkan otomatis)