from models import State
from .base import BaseAgent


class RouterAgent(BaseAgent):
    def run(self, state: State) -> dict:
        message_type = state.get("message_type", "smalltalk")
        if message_type == "route":
            return {"next": "busway_route"}
        elif message_type == "rag":
            return {"next": "rag"}
        else: 
            return {"next": "smalltalk"}
