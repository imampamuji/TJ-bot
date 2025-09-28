from abc import ABC, abstractmethod
from models import State

class BaseAgent(ABC):
    @abstractmethod
    def run(self, state: State) -> dict:
        pass
