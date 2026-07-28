"""
Model Train Protocol (MTP) - A Python package for creating custom Language Model training protocols.

MTP is an open-source protocol for training custom Language Models on Databiomes.
MTP contains all the data that a model is trained on.
"""
from model_train_protocol.v1 import ProtocolFileV1
from model_train_protocol.v1 import TemplateFileV1
from model_train_protocol.v2 import ProtocolFileV2
from model_train_protocol.v2 import TemplateFileV2

__all__ = [
    "ProtocolFileV1",
    "TemplateFileV1",
    "ProtocolFileV2",
    "TemplateFileV2"
]
