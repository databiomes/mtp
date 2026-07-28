"""
Version dispatch for bloom files.

Bloom files come in two incompatible shapes, distinguished by a single field:

* bloom 1.2.x carries a ``state_machine`` boolean and is read by :class:`ProtocolV1`.
* bloom 2.0.0 and later carry a ``model_type`` enum and are read by :class:`ProtocolV2`.

The switch happened at bloom 2.0.0, the first version to carry ``model_type``. Anything below
it predates the change and routes to V1.
"""

from typing import Type, Union

from packaging.version import Version

from model_train_protocol.errors import ProtocolError
from model_train_protocol.v1 import ProtocolV1
from model_train_protocol.v2 import ProtocolV2

# The first bloom version that replaced the `state_machine` boolean with the `model_type` enum.
MODEL_TYPE_BLOOM_VERSION: Version = Version("2.0.0")

# The oldest bloom version any protocol implementation can read.
MINIMUM_BLOOM_VERSION: Version = Version("1.0.0")

# The first bloom version no protocol implementation can read.
MAXIMUM_BLOOM_VERSION: Version = Version("3.0.0")


def get_protocol_for_bloom_version(
    bloom_version: Union[Version, str],
) -> Type[Union[ProtocolV1, ProtocolV2]]:
    """
    Return the protocol class that can read a bloom file of the given version.

    :param bloom_version: The version parsed from the bloom file's ``$schema`` field.
    :return: ``ProtocolV1`` for bloom 1.x, ``ProtocolV2`` for bloom 2.0.0 and later.
    :raises ProtocolError: If no protocol implementation supports the given version.
    """
    if isinstance(bloom_version, str):
        bloom_version = Version(bloom_version)

    if not MINIMUM_BLOOM_VERSION <= bloom_version < MAXIMUM_BLOOM_VERSION:
        raise ProtocolError(
            f"Bloom version {bloom_version} is not supported. Supported versions are "
            f">= {MINIMUM_BLOOM_VERSION} and < {MAXIMUM_BLOOM_VERSION}."
        )

    if bloom_version < MODEL_TYPE_BLOOM_VERSION:
        return ProtocolV1
    return ProtocolV2
