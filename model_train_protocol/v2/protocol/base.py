from abc import ABC, abstractmethod
from typing import Optional

from packaging.version import Version

from model_train_protocol.v2.protocol_file.base import BaseProtocolFile
from model_train_protocol.v2.template_file.base import BaseTemplateFile


class BaseProtocol(ABC):
    """Abstract Base Protocol instance for all protocol versions. Defines the interface and common logic."""

    @property
    @abstractmethod
    def bloom_version(self) -> Version:
        """Returns the Bloom version, e.g. Version("1.2.0")"""

    @abstractmethod
    def from_json(self, *args, **kwargs) -> 'BaseProtocol':
        """Creates a protocol instance from a JSON representation."""

    @abstractmethod
    def validate_protocol(self) -> tuple[bool, Optional[str]]:
        """
        Validates the protocol instance against its schema and rules.

        Validates that the protocol meets all requirements for training.
        Returns: Tuple of (True if valid else False, error message if invalid else None)
        """

    @abstractmethod
    def save(self, name: Optional[str] = None, path: Optional[str] = None):
        """
        Saves the protocol to a JSON file. This file can be submitted to Databiomes for model training.

        :param name: The name of the file (without extension). If None, uses the protocol's name.
        :param path: The directory path where the file will be saved. If None, saves in the current directory.
        """

    @abstractmethod
    def template(self, path: Optional[str] = None):
        """
        Create a template JSON file for the model training protocol.

        The template json file includes example usage and all possible combinations of model inputs and
        outputs based on the defined tokens and instructions.

        :param path: The directory path where the template file will be saved. If None, saves in the current directory.
        """

    @abstractmethod
    def get_protocol_file(self, valid: bool) -> BaseProtocolFile:
        """
        Prepares and returns the ProtocolFile representation of the protocol.

        :return: The ProtocolFile instance representing the protocol.
        """

    @abstractmethod
    def get_template_file(self) -> BaseTemplateFile:
        """
        Prepares and returns the TemplateFile representation of the protocol.

        :return: The TemplateFile instance representing the protocol template.
        """