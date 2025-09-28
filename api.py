from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow import build_graph

# Initialize FastAPI app
app = FastAPI(title="Simple Chatbot API for Telegram")

# Initialize chatbot graph
chatbot_graph = build_graph()

# Simple storage for conversations (in production, use Redis or Database)
conversations = {}

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: str  # Telegram user ID

class ChatResponse(BaseModel):
    response: str

# Helper function
def get_or_create_conversation(user_id: str):
    """Get or create conversation state for user"""
    if user_id not in conversations:
        conversations[user_id] = {
            "messages": [],
            "message_type": None
        }
    return conversations[user_id]

@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Simple chat endpoint for Telegram integration"""
    try:
        # Get user conversation state
        conversation = get_or_create_conversation(request.user_id)
        
        # Add user message
        conversation["messages"].append({
            "role": "user", 
            "content": request.message
        })
        
        # Create state for graph
        state = {
            "messages": conversation["messages"],
            "message_type": conversation["message_type"]
        }
        
        # Get response from chatbot
        result_state = chatbot_graph.invoke(state)
        
        # Extract assistant response
        if result_state.get("messages") and len(result_state["messages"]) > 0:
            last_message = result_state["messages"][-1]
            assistant_response = last_message.content
            
            # Update conversation state
            conversation["messages"] = result_state["messages"]
        else:
            assistant_response = "Maaf, saya tidak dapat memproses pesan Anda."
        
        return ChatResponse(response=assistant_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/chat/{user_id}")
def clear_conversation(user_id: str):
    """Clear conversation history for a user"""
    if user_id in conversations:
        del conversations[user_id]
        return {"message": f"Conversation cleared for user {user_id}"}
    return {"message": "No conversation found for this user"}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)