from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, input_data, **gen_kwargs):
        return


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: GenerativeModel):
        return