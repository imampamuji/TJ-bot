# agents/rag.py

from config import llm
from models import State
from .base import BaseAgent
import os
from vectordb import DocumentPipeline

# Impor schema pesan dari LangChain
from langchain.schema import HumanMessage, SystemMessage

class RAGAgent(BaseAgent):
    def __init__(self):
        """
        Inisialisasi agent.
        Proses pembuatan atau pemuatan vector store dipindahkan ke sini
        agar hanya berjalan sekali saat aplikasi dimulai, bukan setiap kali agent dipanggil.
        """
        self.name = "RAG Agent"
        self.description = "Menjawab pertanyaan faktual TransJakarta dengan persona profesional dan ceria."
        
        print("Inisialisasi RAG Agent...")
        self.pipeline = DocumentPipeline(
            persist_directory="chroma_db",
            chunk_size=500,
            chunk_overlap=50
        )

        # 1. Logika Vector Store dipindahkan ke __init__
        if not os.path.exists(self.pipeline.persist_directory):
            print("Vector store belum ada, membuat baru dari dokumen...")
            self.pipeline.run("data/doc/web_data.md")
        else:
            print("Vector store ditemukan, memuat dari disk...")
            self.pipeline.create_vector_store()
        print("RAG Agent siap digunakan.")

    def run(self, state: State) -> dict:
        """
        Eksekusi agent untuk menjawab pertanyaan pengguna.
        Metode ini sekarang lebih ramping karena hanya fokus pada pencarian dan penjawaban.
        """
        last_message = state["messages"][-1]
        user_query = last_message.content if last_message else ""

        # Cari konteks yang relevan
        context_docs = self.pipeline.vector_store.similarity_search(user_query, k=3)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # 2. Menggunakan struktur prompt System + Human Message yang profesional
        system_prompt = """
        [PERAN & PERSONA]
        Anda adalah "Jaka", petugas pusat informasi virtual TransJakarta. Persona Anda harus selalu:
        - **Profesional**: Memberikan informasi yang akurat dari konteks.
        - **Ramah & Ceria**: Gunakan sapaan "Kak" dan bahasa yang positif serta hangat. Gunakan emoji yang sesuai (misal: 😊, 👍, 🚌).
        - **Sangat Membantu (Helpful)**: Tujuan utama Anda adalah membuat pengguna merasa terlayani dengan baik.

        [TUGAS UTAMA]
        Jawab pertanyaan pengguna SECARA EKSKLUSIF berdasarkan `Konteks` yang diberikan. Anggap `Konteks` adalah satu-satunya sumber kebenaran Anda.

        [ATURAN INTERAKSI]
        1.  **Gunakan Konteks**: Selalu dasarkan jawaban Anda pada informasi yang ada di dalam `Konteks`.
        2.  **Sapa dengan Hangat**: Awali jawaban dengan sapaan ramah jika sesuai.
        3.  **Proaktif Jika Gagal**: Jika jawaban tidak ada di dalam `Konteks`, ikuti alur kerja di bawah ini.

        [ALUR KERJA JIKA INFORMASI TIDAK DITEMUKAN]
        Jika jawaban atas pertanyaan pengguna tidak ada dalam `Konteks`, lakukan langkah-langkah berikut:
        1.  **Akui dengan Sopan**: Mulailah dengan permintaan maaf yang ramah. Contoh: "Mohon maaf, Kak. Untuk informasi spesifik mengenai '{topik pertanyaan}', datanya belum tersedia di sistem Jaka saat ini."
        2.  **Tawarkan Bantuan Alternatif**: Berikan saran yang mungkin bisa membantu. Contoh: "Untuk info yang paling update, Kakak bisa coba cek langsung di website resmi TransJakarta ya."
        3.  **Tutup dengan Positif**: Akhiri dengan tawaran untuk membantu pertanyaan lain. Contoh: "Apakah ada hal lain yang bisa Jaka bantu, Kak? 😊"
        """

        user_input = f"""
        Konteks:
        {context_text}
        ---
        Pertanyaan:
        {user_query}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        # 3. Menggunakan metode .invoke yang lebih modern
        response = llm.invoke(messages)
        answer = response.content

        return {
            "messages": [
                {"role": "assistant", "content": answer}
            ]
        }