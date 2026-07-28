from abc import ABC


class BaseProtocolFile(ABC):
    """Base ProtocolFile class for different model_train_protocol versions."""

    def to_json(self) -> dict:
        """Convert the protocol file to a JSON-serializable dictionary."""
        raise NotImplementedError("Subclasses must implement to_json()")