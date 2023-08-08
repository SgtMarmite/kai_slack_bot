from abc import ABC, abstractmethod


class CommonClient(ABC):
    """
    Declares default AIClient behaviour
    """

    @abstractmethod
    def infer(self, model_name: str, prompt: str, **model_options):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass


class AIClientException(Exception):
    pass
