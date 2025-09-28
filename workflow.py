from langgraph.graph import StateGraph, START, END
from models import State
from agents.classifier import ClassifierAgent
from agents.busway_route import BuswayRouteAgent
from agents.router import RouterAgent
from agents.small_talk import SmalltalkAgent
from agents.rag import RAGAgent


def build_graph():
    graph_builder = StateGraph(State)

    # Inisialisasi agent
    classifier = ClassifierAgent()
    busway_route = BuswayRouteAgent()
    router = RouterAgent()
    smalltalk = SmalltalkAgent()
    rag = RAGAgent()

    # Tambahkan node (pakai method .run)
    graph_builder.add_node("classifier", classifier.run)
    graph_builder.add_node("busway_route", busway_route.run)
    graph_builder.add_node("router", router.run)
    graph_builder.add_node("smalltalk", smalltalk.run )
    graph_builder.add_node("rag", rag.run)

    # Transisi
    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {
        "smalltalk": "smalltalk", 
         "busway_route": "busway_route",
         "rag": "rag"
        } 
    )
    graph_builder.add_edge("busway_route", END)
    graph_builder.add_edge("smalltalk", END)
    graph_builder.add_edge("rag", END)

    return graph_builder.compile()
