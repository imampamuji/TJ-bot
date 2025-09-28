import os
from dotenv import load_dotenv

from config import llm

# LangChain imports
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate


class DocumentPipeline:
    """
    Pipeline untuk memuat dokumen, memotong teks,
    membuat embedding, dan menyimpannya ke Chroma VectorStore.
    """

    def __init__(self,
                 collection_name: str = "default_collection",
                 persist_directory: str = "./chroma_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 200,
                 chunk_overlap: int = 50):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # lazy init
        self.docs: list[Document] = []
        self.chunks: list[Document] = []
        self.embedding = None
        self.vector_store = None

    # -------------------------------
    # 1. Document Loader
    # -------------------------------
    def load_markdown(self, path: str):
        loader = UnstructuredMarkdownLoader(path, mode="single")
        self.docs = loader.load()
        print(f"Loaded {len(self.docs)} documents from {path}")
        return self.docs

    # -------------------------------
    # 2. Text Splitter
    # -------------------------------
    def split_text(self, documents: list[Document] | None = None):
        if documents is None:
            documents = self.docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n"],
        )
        self.chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} docs into {len(self.chunks)} chunks.")
        return self.chunks

    # -------------------------------
    # 3. Embedding Model
    # -------------------------------
    def init_embedding(self):
        self.embedding = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name
        )
        return self.embedding

    # -------------------------------
    # 4. Vector Store
    # -------------------------------
    def create_vector_store(self, chunks: list[Document] | None = None):
        if self.embedding is None:
            self.init_embedding()
        if chunks is None:
            chunks = self.chunks

        # Cek apakah folder sudah ada
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory} ...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_directory
            )
        else:
            print(f"Creating new vector store at {self.persist_directory} ...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_directory
            )
            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Stored {len(chunks)} chunks into {self.persist_directory}")

        return self.vector_store


    # -------------------------------
    # 5. Full pipeline
    # -------------------------------
    def run(self, markdown_path: str):
        self.load_markdown(markdown_path)
        self.split_text()
        self.init_embedding()
        return self.create_vector_store()


if __name__ == "__main__":
    # contoh pemakaian
    pipeline = DocumentPipeline(
        collection_name="bus",
        persist_directory="./chroma_b",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=50
    )

    vector_store = pipeline.run("data/doc/web_data.md")

    q = "tarif berapa?"
    results = vector_store.similarity_search_with_relevance_scores(q, k=5)
    for r, score in results:
        print(f"[Score: {score:.2f}] {r.page_content[:150]}...")

        # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    
#     PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
# {context}
#  - -
# Answer the question based on the above context: {question}
# """
    # System Prompt Opsi 2 (Sangat Direkomendasikan)
    system_prompt = f"""
    [PERAN ANDA]
    Kamu adalah Asisten Faktual TransJakarta. Kamu BUKAN chatbot percakapan.

    [TUGAS UTAMA]
    Tugasmu adalah menjawab pertanyaan pengguna SECARA EKSKLUSIF berdasarkan `Konteks` yang diberikan. Anggap `Konteks` adalah satu-satunya sumber kebenaran.

    [ATURAN KETAT]
    1.  **JAWAB HANYA DARI KONTEKS**: Jangan pernah menggunakan pengetahuan umum atau informasi dari luar `Konteks`. Jawabanmu harus 100% didasarkan pada teks yang disediakan.
    2.  **JANGAN BERASUMSI**: Jika konteks tidak secara eksplisit menyatakan sesuatu, maka kamu tidak mengetahuinya. Jangan mencoba menyimpulkan atau menebak.
    3.  **JAWABAN JIKA TIDAK ADA**: Jika `Konteks` tidak mengandung jawaban atas pertanyaan pengguna, jawab secara jujur dan lugas. Gunakan frasa seperti: "Maaf, informasi mengenai '{q}' tidak ditemukan dalam data yang saya miliki."
    4.  **RINGKAS DAN TEPAT**: Berikan jawaban yang singkat, jelas, dan langsung merujuk pada pertanyaan pengguna.

    """

    # Struktur user message yang lebih baik untuk memisahkan konteks dan query
    user_content = f"""
    Berdasarkan Konteks di bawah ini:
    ---
    {context_text}
    ---
    Jawab pertanyaan berikut: {q}
    """
    
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(system_prompt + user_content)
        
    prompt = prompt_template.format(context=context_text, question=q)
    
    # Initialize OpenAI chat model
    model = llm

    # Generate response text based on the prompt
    response_text = model.predict(prompt)
    
    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # return formatted_response, response_text
    
    print(formatted_response)