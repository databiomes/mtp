from packaging.version import Version

from model_train_protocol.v2.protocol_file.protocol_version import PROTOCOL_VERSION
from model_train_protocol.v2.template_file.template_version import TEMPLATE_VERSION


def get_default_protocol_version() -> Version:
    """Get the default protocol version."""
    return Version(PROTOCOL_VERSION)


def get_default_template_version() -> Version:
    """Get the default template version"""
    return Version(TEMPLATE_VERSION)


from packaging import version


def parse_bloom_file_version(bloom_file: dict) -> version.Version:
    """Parse the version from a Bloom file."""
    if "$schema" not in bloom_file:
        raise ValueError("Bloom file is missing '$schema' field.")
    schema_url: str = bloom_file["$schema"]
    filename = schema_url.split("/")[-1]  # Get the last part of URL

    if "bloom_" in filename:
        # New format: bloom_1_2_0.json -> 1_2_0 -> 1.2.0
        version_part = filename.removeprefix("bloom_").removesuffix(".json")
        schema_version: str = version_part.replace("_", ".")
    else:
        raise ValueError(f"Could not parse version from schema URL: {schema_url}")

    return version.parse(schema_version)


def parse_template_file_version(template_file: dict) -> version.Version:
    """Parse the version from a Template file."""
    if "$schema" not in template_file:
        raise ValueError("Template file is missing '$schema' field.")
    schema_url: str = template_file["$schema"]
    # Extract version from URL like 'https://mtp.schemas.databiomes.com/v1/template_1_2_0.json'
    filename = schema_url.split("/")[-1]  # 'template_1_2_0.json'
    version_part = filename.replace("template_", "").replace(".json", "")  # '1_2_0'
    schema_version: str = version_part.replace("_", ".")  # '1.2.0'
    return version.parse(schema_version)
