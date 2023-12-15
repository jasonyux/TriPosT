from abc import ABC, abstractmethod

class ScriptedVerifier(ABC):
    @abstractmethod
    def verify_rationale(self, data: dict) -> str :
        raise NotImplementedError