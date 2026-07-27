"""Shared fixtures and schema factories for the dsl-spa test suite.

The field definitions these helpers emit mirror ``PipelineField.generate_schema``
in ``dsl_spa.utils.schema`` so that tests stay aligned with the real builder.
"""

import copy

import pytest

from dsl_spa.pipeline.pipeline import Pipeline

# Sentinel distinguishing "no default declared" from "default is None".
_NO_DEFAULT = object()


def field(
    name: str,
    field_type: str = "string",
    required: bool = False,
    description: str = "A test field",
    default=_NO_DEFAULT,
) -> dict:
    """Builds a single field definition dict.

    The ``default`` key is omitted entirely unless a default is supplied, which is
    what distinguishes a field with no default from one defaulting to a falsy value.
    """
    definition = {
        "name": name,
        "type": field_type,
        "required": required,
        "description": description,
    }
    if default is not _NO_DEFAULT:
        definition["default"] = default
    return definition


def schema(sections: dict, name: str = "Test Pipeline") -> dict:
    """Wraps a ``fields`` tree in the minimal surrounding pipeline schema."""
    return {"pipeline_name": name, "fields": sections}


def make_pipeline(fields_input: dict, sections: dict, name: str = "Test Pipeline") -> Pipeline:
    """Builds a Pipeline from a fields input and a ``fields`` schema tree.

    Both arguments are deep-copied so that Pipeline's in-place mutation of its
    input (BUG-10) cannot leak between tests sharing a fixture.
    """
    return Pipeline(copy.deepcopy(fields_input), schema(copy.deepcopy(sections), name), {})


@pytest.fixture
def documented_sections() -> dict:
    """The ``fields`` tree from docs/Creating_a_Pipeline_Schema.md lines 44-78.

    Tests using this fixture double as documentation-conformance checks.
    """
    return {
        "base": {
            "customer_name": field(
                "customer_name", "string", True, "Name of the Customer"
            ),
        },
        "data_filters": {
            "minimum_amount": field(
                "minimum_amount",
                "float",
                False,
                "Minimum amount to be included in search",
                default=10000.0,
            ),
            "earliest_date": field(
                "earliest_date",
                "datetime",
                False,
                "Earliest date to be included in search",
            ),
            "latest_date": field(
                "latest_date",
                "datetime",
                False,
                "Latest date to be included in search",
            ),
        },
    }
