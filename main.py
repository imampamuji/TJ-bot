import gradio as gr
from workflow import build_graph

class ChatbotInterface:
    def __init__(self):
        self.graph = build_graph()
        self.state = {"messages": [], "message_type": None}
    
    def chat_function(self, message, history):
        """
        Fungsi untuk memproses pesan chat
        Args:
            message: pesan dari user
            history: riwayat percakapan dalam format Gradio
        Returns:
            tuple: ("", updated_history) - pesan kosong dan history yang diupdate
        """
        # Reset state jika ini percakapan baru
        if not history:
            self.state = {"messages": [], "message_type": None}
        
        # Tambahkan pesan user ke state
        self.state["messages"] = self.state.get("messages", []) + [
            {"role": "user", "content": message}
        ]
        
        try:
            # Jalankan graph untuk mendapatkan response
            self.state = self.graph.invoke(self.state)
            
            # Ambil response terakhir dari assistant
            if self.state.get("messages") and len(self.state["messages"]) > 0:
                last_message = self.state["messages"][-1]
                assistant_response = last_message.content
            else:
                assistant_response = "Maaf, terjadi kesalahan dalam memproses pesan Anda."
        
        except Exception as e:
            assistant_response = f"Error: {str(e)}"
        
        # Update history dengan format Gradio
        history.append([message, assistant_response])
        
        return "", history
    
    def clear_chat(self):
        """Fungsi untuk membersihkan chat dan reset state"""
        self.state = {"messages": [], "message_type": None}
        return [], []
    
    def create_interface(self):
        """Membuat interface Gradio"""
        
        with gr.Blocks(
            title="My Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 800px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 My Chatbot")
            gr.Markdown("Selamat datang! Silakan ajukan pertanyaan atau mulai percakapan.")
            
            chatbot = gr.Chatbot(
                label="Percakapan",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Pesan Anda",
                    placeholder="Ketik pesan Anda di sini...",
                    lines=2,
                    max_lines=5,
                    scale=4
                )
                send_btn = gr.Button("Kirim", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Bersihkan Chat", variant="secondary")
            
            # Event handlers
            msg_input.submit(
                fn=self.chat_function,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=True
            )
            
            send_btn.click(
                fn=self.chat_function,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=True
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, msg_input],
                queue=False
            )
            
            # Contoh percakapan
            gr.Examples(
                examples=[
                    ["Saya ingin ke Kalideres dari terminal Kampung Rambutan"],
                    ["Bisakah kamu membantu saya?"],
                    ["Apa yang bisa kamu lakukan?"]
                ],
                inputs=msg_input
            )
        
        return interface

def run_chatbot_gradio():
    """Menjalankan chatbot dengan interface Gradio"""
    chatbot_interface = ChatbotInterface()
    interface = chatbot_interface.create_interface()
    
    # Launch interface
    interface.launch(
        server_name="0.0.0.0",  # Bisa diakses dari network
        server_port=7860,       # Port default Gradio
        share=True,            # Set True jika ingin public link
        debug=True,             # Set False untuk production
        show_error=True
    )


if __name__ == "__main__":
    
    # Opsi 1: Menggunakan class (lebih terstruktur)
    run_chatbot_gradio()
    
    # Opsi 2: Implementasi sederhana
    # simple_gradio_chatbot().launch()