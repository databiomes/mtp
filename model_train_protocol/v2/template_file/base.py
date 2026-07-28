from abc import ABC


class BaseTemplateFile(ABC):
    """Base TemplateFile class for different model_train_protocol versions."""

    def to_json(self) -> dict:
        """Convert the template file to a JSON-serializable dictionary."""
        raise NotImplementedError("Subclasses must implement to_json()")