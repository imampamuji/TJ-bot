from config import llm
from models import State, MessageClassifier
from .base import BaseAgent


# System Prompt Opsi 2 (Sangat Direkomendasikan)
system_prompt = """
[TUGAS ANDA]
Anda adalah sebuah AI classifier yang sangat akurat. Tugas utama Anda adalah mengklasifikasikan pesan pengguna ke dalam SATU dari tiga kategori berikut: `route`, `rag`, atau `smalltalk`.
Jawaban Anda HARUS hanya berupa SATU KATA nama kategori tersebut dan tidak boleh ada penjelasan tambahan.

[DEFINISI KATEGORI & ATURAN]

1.  Kategori: `route`
    -   **Aturan**: Gunakan kategori ini HANYA DAN EKSKLUSIF JIKA pesan pengguna secara spesifik menanyakan tentang **perjalanan dari satu titik asal (A) ke satu titik tujuan (B)**. Pesan harus memiliki niat untuk berpindah tempat.
    -   **Kata Kunci Umum**: "dari ... ke ...", "bagaimana cara ke ...", "rute menuju ...", "perjalanan ke ...", "naik apa ke ...".
    -   **Contoh Pesan**:
        -   "Dari Lebak Bulus ke Bundaran HI"
        -   "Gimana caranya ke Ragunan dari Kampung Rambutan?"
        -  "Rute menuju Monas dari Blok M"
        -   "Naik apa ke Stasiun Sudirman dari Harmoni?"
        -   "Bagaimana perjalanan ke Bandara Soekarno Hatta dari Kota?"
        
2.  Kategori: `rag`
    -   **Aturan**: Gunakan kategori ini jika pesan pengguna adalah **pertanyaan yang mencari FAKTA atau INFORMASI SPESIFIK** tentang layanan TransJakarta, yang BUKAN merupakan permintaan rute A ke B.
    -   **Aturan Pengecualian**: Pertanyaan tentang lokasi satu halte saja (tanpa tujuan) masuk ke kategori ini, bukan `route`.
    -   **Kata Kunci Umum**: "berapa", "apa", "di mana", "kapan", "apakah", "info".
    -   **Contoh Pesan**:
        -   "Berapa harga tiket TransJakarta sekarang?"
        -   "Jam operasional busway sampai jam berapa?"
        -   "Apakah anak di bawah 5 tahun bayar?"
        -   "Di mana halte busway yang paling dekat dengan Monas?"  (Ini `rag`, bukan `route` karena tidak ada titik A)
        -   "Bus koridor 13 lewat mana saja?" (Ini `rag`, karena menanyakan info koridor, bukan perjalanan A ke B)

3.  Kategori: `smalltalk`
    -   **Aturan**: Gunakan kategori ini untuk semua interaksi yang bersifat **sosial, sapaan, basa-basi, atau komentar umum**. Pesan ini tidak mengandung permintaan informasi yang bisa dijawab.
    -   **Aturan Pengecualian**: Ini adalah kategori "catch-all" untuk semua yang tidak cocok dengan `route` atau `rag`.
    -   **Contoh Pesan**:
        -   "Halo, selamat pagi"
        -   "Terima kasih banyak ya"
        -   "Oke, paham"
        -   "Wah, Jakarta hari ini panas banget"

"""

class ClassifierAgent(BaseAgent):
    def run(self, state: State) -> dict:
        last_message = state["messages"][-1]
        classifier_llm = llm.with_structured_output(MessageClassifier)
        result = classifier_llm.invoke([
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": last_message.content}
        ])
        print(f"[Classifier] Kategori terdeteksi: {result.message_type}")
        return {"message_type": result.message_type}
      