"""
Tests for bloom version dispatch between ProtocolV1 and ProtocolV2.

Bloom 1.x carries a `state_machine` boolean; bloom 2.0.0 onwards carries a `model_type`
enum. These are incompatible, so each protocol version must accept only its own format and
reject the other with a clear error rather than failing somewhere deeper.
"""

import pytest
from model_train_protocol_schemas.utils import get_example_bloom_file
from packaging.version import Version

from model_train_protocol.errors import ProtocolError
from model_train_protocol.v1 import ProtocolV1
from model_train_protocol.v2 import ProtocolV2
from model_train_protocol.versioning import (
    MODEL_TYPE_BLOOM_VERSION,
    get_protocol_for_bloom_version,
)


class TestBloomVersionDispatch:
    """Version -> protocol class routing."""

    @pytest.mark.parametrize("bloom_version", ["1.0.0", "1.2.0", "1.2.1", "1.2.99"])
    def test_pre_model_type_versions_route_to_v1(self, bloom_version):
        assert get_protocol_for_bloom_version(bloom_version) is ProtocolV1

    @pytest.mark.parametrize("bloom_version", ["2.0.0", "2.4.1"])
    def test_model_type_versions_route_to_v2(self, bloom_version):
        assert get_protocol_for_bloom_version(bloom_version) is ProtocolV2

    def test_boundary_is_the_first_model_type_version(self):
        """2.0.0 is where `state_machine` became `model_type`, so it is the switch point."""
        assert MODEL_TYPE_BLOOM_VERSION == Version("2.0.0")
        assert get_protocol_for_bloom_version(Version("1.999.999")) is ProtocolV1
        assert get_protocol_for_bloom_version(MODEL_TYPE_BLOOM_VERSION) is ProtocolV2

    @pytest.mark.parametrize("bloom_version", ["0.9.0", "3.0.0", "4.1.2"])
    def test_unsupported_versions_raise(self, bloom_version):
        with pytest.raises(ProtocolError, match="not supported"):
            get_protocol_for_bloom_version(bloom_version)

    def test_accepts_version_objects_and_strings(self):
        assert get_protocol_for_bloom_version("2.0.0") is get_protocol_for_bloom_version(
            Version("2.0.0")
        )


class TestProtocolsAcceptOnlyTheirOwnFormat:
    """Each protocol reads its own bloom format and rejects the other's."""

    @pytest.mark.parametrize("bloom_version", ["1.2.0", "1.2.1"])
    def test_v1_reads_state_machine_blooms(self, bloom_version):
        bloom = get_example_bloom_file(Version(bloom_version))
        assert "state_machine" in bloom and "model_type" not in bloom

        template = ProtocolV1.from_json(bloom).get_template_file().to_json()

        assert "template_1_2_0.json" in template["$schema"]
        assert template["state_machine"] is bloom["state_machine"]
        assert "model_type" not in template

    def test_v2_reads_model_type_blooms(self):
        bloom = get_example_bloom_file(Version("2.0.0"))
        assert "model_type" in bloom and "state_machine" not in bloom

        template = ProtocolV2.from_json(bloom).get_template_file().to_json()

        assert "template_2_0_0.json" in template["$schema"]
        assert template["model_type"] == bloom["model_type"]
        assert "state_machine" not in template

    @pytest.mark.parametrize("bloom_version", ["1.2.0", "1.2.1"])
    def test_v2_rejects_state_machine_blooms(self, bloom_version):
        bloom = get_example_bloom_file(Version(bloom_version))
        with pytest.raises(ProtocolError, match="model_type"):
            ProtocolV2.from_json(bloom)

    def test_v1_rejects_model_type_blooms(self):
        bloom = get_example_bloom_file(Version("2.0.0"))
        with pytest.raises(ProtocolError, match="state_machine"):
            ProtocolV1.from_json(bloom)


class TestVersionedFieldRequirementsArePinned:
    """
    Each protocol pins its own required-field list.

    Importing them from model_train_protocol_schemas would mean validating against whichever
    bloom version the schemas package happens to ship, which is what broke V1 when the
    schemas package moved to `model_type`.
    """

    def test_v1_requires_state_machine_not_model_type(self):
        from model_train_protocol.v1.protocol.protocol_v1 import BLOOM_V1_REQUIRED_FIELDS

        assert "state_machine" in BLOOM_V1_REQUIRED_FIELDS
        assert "model_type" not in BLOOM_V1_REQUIRED_FIELDS

    def test_v2_requires_model_type_not_state_machine(self):
        from model_train_protocol.v2.protocol.protocol_v2 import BLOOM_V2_REQUIRED_FIELDS

        assert "model_type" in BLOOM_V2_REQUIRED_FIELDS
        assert "state_machine" not in BLOOM_V2_REQUIRED_FIELDS

    def test_each_version_emits_its_own_schema_version(self):
        from model_train_protocol.v1.template_file.template_version import (
            TEMPLATE_VERSION as V1_TEMPLATE_VERSION,
        )
        from model_train_protocol.v2.template_file.template_version import (
            TEMPLATE_VERSION as V2_TEMPLATE_VERSION,
        )

        assert V1_TEMPLATE_VERSION == "1.2.0"
        assert V2_TEMPLATE_VERSION == "2.0.0"
