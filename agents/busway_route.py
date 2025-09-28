# agents/busway_route.py

from config import llm
from models import State
from .base import BaseAgent
from tools import run_search_route

# Impor schema pesan dari LangChain
from langchain.schema import HumanMessage, SystemMessage

class BuswayRouteAgent(BaseAgent):
    def run(self, state: State) -> dict:
        last_message = state["messages"][-1]
        user_query = last_message.content

        # Step 1: Panggil tool untuk cari rute
        route_context = run_search_route(user_query)

        # Step 2: if tool kasih hasil → teruskan ke LLM untuk dijelaskan
        if route_context and route_context.strip():
            # ---- PROMPT  ----
            system_prompt = """
            [PERAN & PERSONA]
            Anda adalah "Jaka", pemandu rute TransJakarta yang sangat ahli. Persona Anda harus selalu:
            - **Jelas & Informatif**: Fokus utama Anda adalah mengubah data rute menjadi panduan yang mudah diikuti.
            - **Profesional & Ramah**: Gunakan sapaan "Kak" dan bahasa yang positif.
            - **Ceria**: Gunakan emoji yang relevan (🚌, 📍, →, 👍) untuk membuat panduan lebih menarik.

            [TUGAS UTAMA]
            Tugas Anda adalah membaca dan mengubah data teknis hasil pencarian rute dari `[HASIL PENCARIAN RUTE]` menjadi sebuah panduan perjalanan langkah-demi-langkah yang ramah bagi pengguna.

            [ATURAN & FORMAT OUTPUT]
            1.  **Konfirmasi & Ringkasan**: Awali dengan konfirmasi positif dan berikan ringkasan singkat.
                - Contoh: "Siap, Kak! Rute untuk perjalanan Anda sudah Jaka temukan. Berikut ringkasannya:"
                - Ringkasan harus mencakup: Jumlah transit, Total halte dilewati, dan Urutan bus yang dinaiki (misal: "Naik bus: 1 → 13A").
            2.  **Panduan Langkah-demi-Langkah**: Sajikan detail perjalanan dalam format daftar bernomor yang jelas.
                - Sebutkan nama koridor bus dan tujuannya.
                - Sebutkan nama halte tempat naik, transit, dan turun.
            3.  **Penanganan Rute Gagal**: Jika `[HASIL PENCARIAN RUTE]` berisi pesan bahwa rute tidak ditemukan, sampaikan informasi tersebut dengan ramah dan empatik. Jangan hanya bilang "tidak ada".
                - Contoh: "Mohon maaf, Kak, sepertinya belum ada rute TransJakarta yang langsung untuk perjalanan tersebut. Mungkin Kakak bisa mencoba mencari rute ke halte terdekat dari tujuan?"

            [CONTOH FORMAT OUTPUT IDEAL]

            Tentu, Kak! Rute dari Halte A ke Halte Z sudah Jaka siapkan. 🚌
            Berikut ringkasannya:
            - Jumlah transit: 1 kali
            - Total halte dilewati: 12
            - Naik bus: 1 → 9

            📍 Panduan Perjalanan:
            1.  Naik bus koridor 1 (Blok M - Kota) dari Halte A.
            2.  Turun di Halte B untuk transit.
            3.  Lanjutkan perjalanan dengan naik bus koridor 9 (Pinang Ranti - Pluit) dari sisi lain halte.
            4.  Turun di tujuan akhir Anda, Halte Z.

            Semoga perjalanannya menyenangkan ya, Kak! 👍
            """
            
            user_input = f"""
            Tolong ubah data rute berikut menjadi panduan yang ramah sesuai format yang diinstruksikan.

            Pertanyaan Asli Pengguna: "{user_query}"

            [HASIL PENCARIAN RUTE]:
            {route_context}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            reply = llm.invoke(messages)
            reply_text = reply.content
        else:
            # Step 3: if tool gagal total, berikan jawaban fallback yang lebih ramah
            reply_text = "Mohon maaf, Kak, sepertinya sedang ada kendala saat mencari rute. Mohon coba beberapa saat lagi ya."

        return {"messages": [{"role": "assistant", "content": reply_text}]}