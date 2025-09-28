import os
import requests
import zipfile
import pandas as pd
from collections import defaultdict, deque
from config import llm
import chromadb
from config import embedding_model

import pandas as pd
import os
from rapidfuzz import process
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

 
class GTFSDataManager:
    """
    Kelas untuk mengelola data GTFS: mengunduh, mengekstrak, memuat, dan memproses.
    """
    def __init__(self, gtfs_url="https://gtfs.transjakarta.co.id/files/file_gtfs.zip", data_dir="data/gtfs"):
        self.gtfs_url = gtfs_url
        self.data_dir = data_dir
        self.zip_path = os.path.join(self.data_dir, "file_gtfs.zip")
        
        # Atribut untuk menyimpan dataframes dan data yang telah diproses
        self.stops = None
        self.routes = None
        self.trips = None
        self.stop_times = None
        
        self.stop_name_to_ids = defaultdict(list)
        self.trip_stops_dict = {}
        self.stop_to_trips = defaultdict(set)
        self.trip_to_route = {}
        self.route_to_name = {}
        self.route_to_short_name = {}
        self.stop_id_to_name = {}

    def download_and_extract(self):
        """Mengunduh dan mengekstrak file GTFS dari URL."""
        os.makedirs(self.data_dir, exist_ok=True)

        print(f"Downloading GTFS data from {self.gtfs_url} ...")
        try:
            # Nonaktifkan verifikasi SSL jika perlu, tapi lebih baik pasang sertifikat yang benar
            response = requests.get(self.gtfs_url, stream=True, verify=False)
            response.raise_for_status()  # Akan melempar error jika status code bukan 2xx
            
            with open(self.zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded to {self.zip_path}")

            print("Extracting zip file...")
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"✅ Extracted to {self.data_dir}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download GTFS: {e}")
            raise
        finally:
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)
                print(f"Removed zip file {self.zip_path}")
    
    def load_and_preprocess_data(self):
        """Memuat file GTFS ke dalam DataFrame dan melakukan preprocessing."""
        print("Loading and preprocessing GTFS data...")
        try:
            self.stops = pd.read_csv(os.path.join(self.data_dir, "stops.txt"))
            self.routes = pd.read_csv(os.path.join(self.data_dir, "routes.txt"))
            self.trips = pd.read_csv(os.path.join(self.data_dir, "trips.txt"))
            self.stop_times = pd.read_csv(os.path.join(self.data_dir, "stop_times.txt"))
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}. Please run download_and_extract first.")
            return

        # Preprocessing
        for _, row in self.stops.iterrows():
            self.stop_name_to_ids[row['stop_name'].lower()].append(row['stop_id'])

        stop_times_sorted = self.stop_times.sort_values(['trip_id', 'stop_sequence'])
        for trip_id, group in stop_times_sorted.groupby('trip_id'):
            self.trip_stops_dict[trip_id] = group['stop_id'].tolist()

        for trip_id, stops_list in self.trip_stops_dict.items():
            for stop_id in stops_list:
                self.stop_to_trips[stop_id].add(trip_id)

        self.trip_to_route = dict(zip(self.trips['trip_id'], self.trips['route_id']))
        self.route_to_name = dict(zip(self.routes['route_id'], self.routes['route_long_name']))
        self.route_to_short_name = dict(zip(self.routes['route_id'], self.routes['route_short_name']))
        self.stop_id_to_name = dict(zip(self.stops['stop_id'], self.stops['stop_name']))
        
        print("✅ Data preprocessing complete.")


class RouteFinder:
    """
    Kelas untuk menemukan rute bus antara dua halte menggunakan data GTFS yang telah diproses.
    """
    def __init__(self, data_manager: GTFSDataManager):
        self.data = data_manager

    def find_stop_ids(self, stop_name):
        """Mencari stop_id berdasarkan nama halte (fuzzy matching)."""
        stop_name_lower = stop_name.lower()
        if stop_name_lower in self.data.stop_name_to_ids:
            return self.data.stop_name_to_ids[stop_name_lower]
        
        matches = [id for name, ids in self.data.stop_name_to_ids.items() if stop_name_lower in name for id in ids]
        return matches

    def get_route_transfer_info(self, trips_used, path):
        """Mendapatkan informasi detail rute dan titik transfer."""
        unique_trips = list(dict.fromkeys(trips_used))
        routes_info = []
        transfer_points = []

        for i, trip_id in enumerate(unique_trips):
            route_id = self.data.trip_to_route.get(trip_id)
            if not route_id: continue

            routes_info.append({
                "route_short_name": self.data.route_to_short_name.get(route_id, "N/A"),
                "route_long_name": self.data.route_to_name.get(route_id, "N/A"),
            })

            if i < len(unique_trips) - 1:
                next_trip = unique_trips[i + 1]
                current_trip_stops = set(self.data.trip_stops_dict.get(trip_id, []))
                next_trip_stops = set(self.data.trip_stops_dict.get(next_trip, []))
                
                # Cari titik transfer yang ada di path
                for stop_id in path:
                    if stop_id in current_trip_stops and stop_id in next_trip_stops:
                        transfer_points.append({
                            "stop_name": self.data.stop_id_to_name.get(stop_id, "Unknown"),
                            "from_route": self.data.route_to_short_name.get(route_id, "N/A"),
                            "to_route": self.data.route_to_short_name.get(self.data.trip_to_route.get(next_trip), "N/A"),
                        })
                        break # Ambil titik transfer pertama yang ditemukan di path
        
        return routes_info, transfer_points

    def find_route(self, start_stop_name, end_stop_name, max_transits=2):
        """Mencari rute tercepat (jumlah halte paling sedikit) antara dua halte."""
        start_ids = self.find_stop_ids(start_stop_name)
        end_ids = self.find_stop_ids(end_stop_name)

        if not start_ids: return f"Halte awal '{start_stop_name}' tidak ditemukan."
        if not end_ids: return f"Halte tujuan '{end_stop_name}' tidak ditemukan."

        end_ids_set = set(end_ids)
        queue = deque()
        visited = set()

        for start_id in start_ids:
            for trip_id in self.data.stop_to_trips[start_id]:
                state = (start_id, trip_id, [start_id], [trip_id])
                queue.append(state)
                visited.add((start_id, trip_id))

        while queue:
            current_stop, current_trip, path, trips_used = queue.popleft()

            if current_stop in end_ids_set:
                routes_info, transfer_points = self.get_route_transfer_info(trips_used, path)
                return {
                    "transit_stops": [self.data.stop_id_to_name.get(s, "?") for s in path],
                    "routes_detail": routes_info,
                    "transfer_points": transfer_points,
                    "num_transits": len(set(trips_used)) - 1,
                    "total_stops": len(path)
                }
            
            # Batasi jumlah transit
            if len(set(trips_used)) - 1 >= max_transits:
                # Cek perhentian selanjutnya di trip yang sama sebelum menyerah
                next_stops_in_trip = self.data.trip_stops_dict.get(current_trip, [])
                try:
                    current_idx = next_stops_in_trip.index(current_stop)
                    for next_stop in next_stops_in_trip[current_idx + 1:]:
                         if (next_stop, current_trip) not in visited:
                            visited.add((next_stop, current_trip))
                            queue.append((next_stop, current_trip, path + [next_stop], trips_used))
                except ValueError:
                    pass
                continue # Jangan coba transit lagi jika sudah mencapai batas


            # 1. Lanjutkan dalam trip yang sama
            stops_in_trip = self.data.trip_stops_dict.get(current_trip, [])
            try:
                current_idx = stops_in_trip.index(current_stop)
                for next_stop in stops_in_trip[current_idx + 1:]:
                    if (next_stop, current_trip) not in visited:
                        visited.add((next_stop, current_trip))
                        queue.append((next_stop, current_trip, path + [next_stop], trips_used))
            except ValueError:
                pass

            # 2. Coba transit ke trip lain di halte saat ini
            for new_trip in self.data.stop_to_trips[current_stop]:
                if new_trip != current_trip and (current_stop, new_trip) not in visited:
                    visited.add((current_stop, new_trip))
                    queue.append((current_stop, new_trip, path, trips_used + [new_trip]))
        
        return "Rute tidak ditemukan."

class RouteExtractor:
    def __init__(self, stops_filepath=None):
        """
        Inisialisasi dengan memuat data dan setup chain sekali saja.
        """
        print("Initializing RouteExtractor...")
        # === 1. Load Data (dilakukan sekali saat inisialisasi) ===
        if stops_filepath is None:
            # Build absolute path from this file's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stops_filepath = os.path.join(script_dir, '..', 'data', 'gtfs', 'stops.csv')

        try:
            stops = pd.read_csv(stops_filepath)
            self.stop_names = stops["stop_name"].unique().tolist()
        except FileNotFoundError:
            print(f"Error: File not found at {stops_filepath}")
            self.stop_names = []

        # === 2. Setup LLM dan Parser ===
        # llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash", 
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        
        parser = JsonOutputParser()

        # === 3. Definisikan Prompt dengan instruksi format dari parser ===
        prompt = PromptTemplate(
            template="""Tugas kamu: ekstrak asal (origin) dan tujuan (destination) dari pertanyaan tentang perjalanan TransJakarta.
PENTING: Jangan sertakan kata "halte" dalam jawabanmu, hanya nama lokasinya saja.
Jawab hanya sesuai format yang diminta.

{format_instructions}

Contoh:
- "Bagaimana cara ke Monas dari Blok M?" -> {{"origin": "Blok M", "destination": "Monas"}}
- "Saya mau ke Ragunan, naik dari halte Kampung Rambutan" -> {{"origin": "Kampung Rambutan", "destination": "Ragunan"}}
- "Dari halte Harmoni ke halte Gelora Bung Karno naik apa?" -> {{"origin": "Harmoni", "destination": "Gelora Bung Karno"}}

Ekstrak dari pesan ini:
"{query}"
""",
            input_variables=["query"],
            # Menyertakan instruksi format dari parser secara dinamis
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # === 4. Rangkai menjadi satu chain menggunakan LCEL ===
        self.chain = prompt | llm | parser
        print("Initialization complete.")

    def _fuzzy_match_stop(self, name: str, threshold=80):
        """
        Cari nama halte terdekat dengan fuzzy matching.
        Threshold dinaikkan sedikit untuk akurasi yang lebih baik.
        """
        if not self.stop_names or not name:
            return None
        
        # process.extractOne mengembalikan (match, score, index)
        match, score, _ = process.extractOne(name, self.stop_names)
        return match if score > threshold else None

    def get_route_from_query(self, query: str):
        """
        Mengekstrak dan memvalidasi asal & tujuan dari query pengguna.
        """
        try:
            # Step 1: Jalankan chain untuk parsing (lebih sederhana)
            parsed = self.chain.invoke({"query": query})
            
            # Step 2: Gunakan .get() untuk menghindari KeyError jika LLM gagal
            origin_raw = parsed.get("origin")
            destination_raw = parsed.get("destination")
            
            if not origin_raw or not destination_raw:
                print("Error: LLM did not return 'origin' or 'destination'.")
                return None

            # Step 3: Validasi nama halte dengan fuzzy matching
            origin_validated = self._fuzzy_match_stop(origin_raw)
            destination_validated = self._fuzzy_match_stop(destination_raw)

            return {
                "origin_raw": origin_raw,
                "destination_raw": destination_raw,
                "origin_validated": origin_validated,
                "destination_validated": destination_validated
            }

        except OutputParserException as e:
            # Error jika output LLM bukan JSON yang valid
            print(f"Error parsing LLM output: {e}")
            return None
        except Exception as e:
            # Error lainnya
            print(f"An unexpected error occurred: {e}")
            return None

def print_route_details(result):
    """Fungsi helper untuk menampilkan hasil pencarian rute dengan format yang rapi."""
    if isinstance(result, str):
        print(f"❌ {result}")
        return

    print("✅ Rute ditemukan!")
    print(f"Jumlah transit: {result['num_transits']}")
    print(f"Total halte dilewati: {result['total_stops']}")
    
    # Menampilkan rute bus
    route_names = [r['route_short_name'] for r in result['routes_detail']]
    print(f"Naik bus: {' → '.join(route_names)}")
    
    # Menampilkan panduan perjalanan
    print("\n📍 Panduan Perjalanan:")
    print(f"  1. Naik bus {result['routes_detail'][0]['route_short_name']} ({result['routes_detail'][0]['route_long_name']}) dari {result['transit_stops'][0]}.")
    
    if result['transfer_points']:
        for i, transfer in enumerate(result['transfer_points']):
            print(f"  {i+2}. Turun di halte {transfer['stop_name']} untuk transit.")
            print(f"     Lanjutkan dengan bus {transfer['to_route']} ({result['routes_detail'][i+1]['route_long_name']}).")
    
    print(f"  {len(result['transfer_points']) + 2}. Turun di tujuan akhir Anda, {result['transit_stops'][-1]}.")

def format_route_details_to_string(result):
    """
    Mengubah hasil pencarian rute menjadi satu string tunggal yang terformat rapi.
    """
    # Jika hasil adalah string (pesan error), kembalikan langsung
    if isinstance(result, str):
        return f"❌ {result}"

    # Gunakan list untuk menampung setiap baris teks
    output_lines = []

    output_lines.append("✅ Rute ditemukan!")
    output_lines.append(f"Jumlah transit: {result['num_transits']}")
    output_lines.append(f"Total halte dilewati: {result['total_stops']}")
    
    # Menyiapkan rute bus
    route_names = [r['route_short_name'] for r in result['routes_detail']]
    output_lines.append(f"Naik bus: {' → '.join(route_names)}")
    
    # Menambahkan baris kosong sebelum panduan
    output_lines.append("\n📍 Panduan Perjalanan:")
    
    # Langkah pertama
    start_stop = result['transit_stops'][0]
    start_route_short = result['routes_detail'][0]['route_short_name']
    start_route_long = result['routes_detail'][0]['route_long_name']
    output_lines.append(f"  1. Naik bus {start_route_short} ({start_route_long}) dari {start_stop}.")
    
    # Langkah transit (jika ada)
    if result['transfer_points']:
        for i, transfer in enumerate(result['transfer_points']):
            next_route_short = transfer['to_route']
            next_route_long = result['routes_detail'][i+1]['route_long_name']
            output_lines.append(f"  {i+2}. Turun di halte {transfer['stop_name']} untuk transit.")
            output_lines.append(f"     Lanjutkan dengan bus {next_route_short} ({next_route_long}).")
    
    # Langkah terakhir
    final_step_number = len(result['transfer_points']) + 2
    final_stop = result['transit_stops'][-1]
    output_lines.append(f"  {final_step_number}. Turun di tujuan akhir Anda, {final_stop}.")

    # Gabungkan semua baris menjadi satu string dengan pemisah baris baru
    return "\n".join(output_lines)

def run_search_route(query: str):
    """
    Fungsi utama untuk menjalankan ekstraksi rute dari input pengguna.
    """
    # 1. Inisialisasi dan siapkan data
    gtfs_manager = GTFSDataManager()
    # Hapus komentar di baris berikut jika Anda perlu mengunduh data baru
    # gtfs_manager.download_and_extract()
    gtfs_manager.load_and_preprocess_data()

    # 2. Inisialisasi pencari rute dengan data yang sudah siap
    route_finder = RouteFinder(gtfs_manager)
    
    # Buat instance extractor (setup data & LLM hanya terjadi di sini)
    extractor = RouteExtractor()
    print(extractor.stop_names[:5])  # Cek beberapa nama halte
    
    print("\n" + "="*30)
     
    # q = "Saya mau berangkat dari kampung rambutan ke kalideres, naik bus apa ya?"
    result = extractor.get_route_from_query(query)

    # # Selalu cek jika result tidak None sebelum digunakan
    # if result:
    #     print(f"Query Pengguna: '{q}'")
    #     print(f"  -> Ekstraksi Asal (Raw): '{result['origin_raw']}'")
    #     print(f"  -> Ekstraksi Tujuan (Raw): '{result['destination_raw']}'")
    #     print(f"  -> Validasi Asal: '{result['origin_validated']}'")
    #     print(f"  -> Validasi Tujuan: '{result['destination_validated']}'")

    #     # Cek jika ada halte yang tidak ditemukan
    #     if not result['origin_validated']:
    #         print(f"⚠️ Peringatan: Halte asal '{result['origin_raw']}' tidak ditemukan di data GTFS.")
    #     if not result['destination_validated']:
    #         print(f"⚠️ Peringatan: Halte tujuan '{result['destination_raw']}' tidak ditemukan di data GTFS.")
    # else:
    #     print("Gagal mengekstrak rute dari query.")
    
    
    formatted_string_output = ""
    # Pastikan data berhasil dimuat sebelum mencari rute
    if gtfs_manager.stops is not None:
        hasil_rute = route_finder.find_route(result["origin_raw"], result["destination_raw"])
        print(f"Perjalanan dari {result['origin_raw']} ke {result['destination_raw']}:")
        formatted_string_output = format_route_details_to_string(hasil_rute)
        print(formatted_string_output)
    else:
        print("Data tidak dapat dimuat, pencarian rute dibatalkan.")
    
    return formatted_string_output


###
"""
Tools for RAG
"""
###

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def split_markdown_file(filepath: str, chunk_size=500, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# init Chroma client
chroma_client = chromadb.Client()

# buat collection 
collection = chroma_client.create_collection(name="busway_docs")

def add_to_chroma(chunks):
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"✅ Added {len(chunks)} chunks to ChromaDB")
    
from config import llm

def answer_query(query: str):
    # embed query
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # retrieve dari Chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    context = "\n".join(results["documents"][0])
    
    # kirim ke LLM
    messages = [
        {"role": "system", "content": "You are an assistant that answers based on website content."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    reply = llm.invoke(messages)
    return reply.content




# --- Contoh Penggunaan ---
if __name__ == "__main__":
    run_search_route("Saya mau berangkat dari halte blok m ke monas, naik bus apa ya?")
        
    
