from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class MessageClassifier(BaseModel):
    message_type: Literal["route", "rag", "smalltalk"] = Field(
        ...,
        description="Classify if the message is about route planning, general knowledge (RAG), or small talk."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
