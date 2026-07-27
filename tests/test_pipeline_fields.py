"""Tests for the field-processing logic in the base ``Pipeline`` class.

Fields are the only component present in every dsl-spa pipeline, and
``add_fields_to_clause`` is the substitution engine behind queries, summaries,
visualization titles, CSV column filters and action parameters. Everything here
exercises ``Pipeline`` alone - no connectors, no pandas, no altair.

Tests assert the *documented and intended* behavior. Where the current
implementation is broken, the test is marked ``xfail(strict=True)`` with the
bug it is waiting on. A clean run therefore proves both that the working paths
still work and that every known bug is still present; the moment a bug is fixed
its test flips from ``xfailed`` to a hard ``XPASS`` failure, so this file
doubles as the fix checklist. Run ``pytest -rxX`` to print it.
"""

import copy

import pytest

from conftest import field, make_pipeline, schema
from dsl_spa.pipeline.pipeline import Pipeline, PipelineException


@pytest.fixture
def accessor_pipeline() -> Pipeline:
    """A pipeline with a variety of value shapes for exercising the accessors."""
    return make_pipeline(
        {"s": "v", "n": 5, "flag": False, "nothing": None, "sec": {"n": 1}},
        {"base": {"s": field("s")}},
    )


class TestConstruction:
    def test_stores_pipeline_name(self):
        pipeline = make_pipeline({}, {}, name="My Pipeline")
        assert pipeline.pipeline_name == "My Pipeline"

    def test_stores_schema(self):
        sections = {"base": {"x": field("x")}}
        pipeline = make_pipeline({}, sections)
        assert pipeline.schema["fields"] == sections

    def test_stores_connectors(self):
        connectors = {"csvs": object()}
        pipeline = Pipeline({}, schema({}), connectors)
        assert pipeline.connectors is connectors

    def test_empty_fields_schema_is_accepted(self):
        assert make_pipeline({}, {}).field_dict == {}

    def test_input_fields_not_in_schema_are_preserved(self):
        pipeline = make_pipeline({"extra": "kept"}, {})
        assert pipeline.field_dict["extra"] == "kept"

    def test_documented_example_populates_defaults(self, documented_sections):
        """docs/Creating_a_Pipeline_Schema.md lines 29-78, end to end."""
        pipeline = make_pipeline({"customer_name": "Acme"}, documented_sections)
        assert pipeline.field_dict == {
            "customer_name": "Acme",
            "data_filters": {"minimum_amount": 10000.0},
        }
        assert pipeline.required_fields == ["base.customer_name"]

    def test_documented_example_requires_customer_name(self, documented_sections):
        with pytest.raises(PipelineException):
            make_pipeline({}, documented_sections)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-10: __init__ aliases fields_input_dict instead of copying it, "
        "so defaults and categorical one-hots leak back into the caller's dict",
    )
    def test_does_not_mutate_the_callers_input_dict(self):
        fields_input = {"temperature_scale": "celsius"}
        Pipeline(
            fields_input,
            schema({"base": {"temperature_scale": field("temperature_scale", "categorical")}}),
            {},
        )
        assert fields_input == {"temperature_scale": "celsius"}

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-11: a schema missing 'fields' raises a bare KeyError; "
        "PipelineException exists precisely to distinguish schema errors",
    )
    def test_schema_without_fields_raises_pipeline_exception(self):
        with pytest.raises(PipelineException):
            Pipeline({}, {"pipeline_name": "p"}, {})

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-11: a schema missing 'pipeline_name' raises a bare KeyError",
    )
    def test_schema_without_pipeline_name_raises_pipeline_exception(self):
        with pytest.raises(PipelineException):
            Pipeline({}, {"fields": {}}, {})


class TestCheckIfFieldDefinition:
    def test_all_four_keys_present(self):
        assert Pipeline.check_if_field_definition(field("x")) is True

    def test_optional_default_key_does_not_break_detection(self):
        assert Pipeline.check_if_field_definition(field("x", default="dv")) is True

    def test_unknown_extra_keys_do_not_break_detection(self):
        definition = field("x")
        definition["units"] = "celsius"
        assert Pipeline.check_if_field_definition(definition) is True

    @pytest.mark.parametrize("omitted", ["name", "type", "required", "description"])
    def test_any_missing_required_key_means_not_a_field(self, omitted):
        definition = field("x")
        del definition[omitted]
        assert Pipeline.check_if_field_definition(definition) is False

    def test_empty_dict_is_not_a_field(self):
        assert Pipeline.check_if_field_definition({}) is False

    def test_a_section_of_fields_is_not_a_field(self):
        assert Pipeline.check_if_field_definition({"x": field("x")}) is False

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-11: calls .keys() on the argument, so a non-dict raises "
        "AttributeError instead of returning False",
    )
    def test_non_dict_is_not_a_field(self):
        assert Pipeline.check_if_field_definition("not a dict") is False

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-11: a field missing 'description' is misread as a section, "
        "then recursion into its string values raises AttributeError",
    )
    def test_malformed_field_definition_raises_pipeline_exception(self):
        malformed = {"name": "x", "type": "string", "required": True}
        with pytest.raises(PipelineException):
            make_pipeline({}, {"base": {"x": malformed}})


class TestDefaultValues:
    def test_base_section_default_lands_at_root(self):
        pipeline = make_pipeline({}, {"base": {"x": field("x", default="dv")}})
        assert pipeline.field_dict == {"x": "dv"}

    def test_section_default_lands_inside_the_section(self):
        pipeline = make_pipeline({}, {"sec": {"y": field("y", default="sy")}})
        assert pipeline.field_dict == {"sec": {"y": "sy"}}

    def test_every_default_in_a_section_is_applied(self):
        sections = {
            "sec": {
                "y": field("y", default="sy"),
                "z": field("z", "integer", default=7),
                "w": field("w", "float", default=1.5),
            }
        }
        assert make_pipeline({}, sections).field_dict == {
            "sec": {"y": "sy", "z": 7, "w": 1.5}
        }

    def test_field_without_a_default_produces_no_key(self):
        pipeline = make_pipeline({}, {"base": {"x": field("x")}})
        assert "x" not in pipeline.field_dict

    def test_supplied_value_wins_over_default_at_base(self):
        pipeline = make_pipeline({"x": "given"}, {"base": {"x": field("x", default="dv")}})
        assert pipeline.field_dict["x"] == "given"

    def test_supplied_value_wins_over_default_in_a_section(self):
        sections = {
            "sec": {
                "y": field("y", default="sy"),
                "z": field("z", "integer", default=7),
            }
        }
        pipeline = make_pipeline({"sec": {"y": "given"}}, sections)
        assert pipeline.field_dict["sec"] == {"y": "given", "z": 7}

    @pytest.mark.parametrize(
        "field_type,falsy",
        [("integer", 0), ("float", 0.0), ("string", ""), ("bool", False)],
    )
    def test_falsy_defaults_are_applied(self, field_type, falsy):
        """A falsy default must be applied, not skipped as though absent.

        Each default is paired with a matching declared type: since type coercion
        went live, a falsy default under a mismatched type is legitimately coerced
        (a "string" field defaulting to 0 becomes "0"), which would mask the
        skipped-vs-applied distinction this test exists to check.
        """
        sections = {"base": {"x": field("x", field_type, default=falsy)}}
        pipeline = make_pipeline({}, sections)
        assert "x" in pipeline.field_dict
        assert pipeline.field_dict["x"] == falsy

    def test_default_merges_into_a_partially_populated_section(self):
        pipeline = make_pipeline(
            {"sec": {"other": "supplied"}},
            {"sec": {"y": field("y", default="sy")}},
        )
        assert pipeline.field_dict["sec"] == {"other": "supplied", "y": "sy"}

    def test_get_default_values_is_callable_without_a_pipeline(self):
        """It is declared without ``self`` and used as a static helper."""
        result = Pipeline.get_default_values({"base": {"x": field("x", default="dv")}})
        assert result == {"x": "dv"}

    def test_get_default_values_handles_two_level_sections(self):
        sections = {"outer": {"inner": {"a": field("a", default="dv")}}}
        assert Pipeline.get_default_values(sections) == {"outer": {"inner": {"a": "dv"}}}

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-14: writes default_dict[key] = {} then default_dict[d['name']], "
        "leaving a spurious empty dict behind when the dict key differs from 'name'",
    )
    def test_get_default_values_keys_defaults_consistently(self):
        sections = {"sec": {"keyname": field("realname", default="dv")}}
        assert Pipeline.get_default_values(sections) == {"sec": {"realname": "dv"}}

    def test_two_level_section_default_is_applied(self):
        """Regression: fill_categorical_values did 'root + key' (list + str) instead
        of 'root + [key]', so a two-level section raised TypeError at construction."""
        sections = {"outer": {"inner": {"a": field("a", default="dv")}}}
        pipeline = make_pipeline({}, sections)
        assert pipeline.field_dict == {"outer": {"inner": {"a": "dv"}}}


class TestCategoricalValues:
    """The documented example: base.temperature_scale is celsius or fahrenheit."""

    SECTIONS = {"base": {"temperature_scale": field("temperature_scale", "categorical")}}

    def test_selected_value_becomes_a_true_one_hot_field(self):
        pipeline = make_pipeline({"temperature_scale": "celsius"}, self.SECTIONS)
        assert pipeline.field_dict["temperature_scale_celsius"] is True

    def test_unselected_value_gets_no_field_at_all(self):
        pipeline = make_pipeline({"temperature_scale": "celsius"}, self.SECTIONS)
        assert "temperature_scale_fahrenheit" not in pipeline.field_dict

    def test_original_field_value_is_preserved(self):
        pipeline = make_pipeline({"temperature_scale": "celsius"}, self.SECTIONS)
        assert pipeline.field_dict["temperature_scale"] == "celsius"

    def test_one_hot_is_reachable_through_get_field(self):
        pipeline = make_pipeline({"temperature_scale": "celsius"}, self.SECTIONS)
        assert pipeline.get_field("base.temperature_scale_celsius") is True
        assert pipeline.check_for_field("base.temperature_scale_celsius") is True

    def test_absent_categorical_produces_no_one_hot(self):
        assert make_pipeline({}, self.SECTIONS).field_dict == {}

    def test_categorical_supplied_by_default_is_still_one_hot_encoded(self):
        """Defaults are populated before categoricals, so a default one-hots too."""
        sections = {
            "base": {
                "temperature_scale": field(
                    "temperature_scale", "categorical", default="celsius"
                )
            }
        }
        pipeline = make_pipeline({}, sections)
        assert pipeline.field_dict == {
            "temperature_scale": "celsius",
            "temperature_scale_celsius": True,
        }

    def test_section_categorical_one_hots_inside_its_section(self):
        pipeline = make_pipeline(
            {"sec": {"unit": "kg"}},
            {"sec": {"unit": field("unit", "categorical")}},
        )
        assert pipeline.field_dict["sec"]["unit_kg"] is True
        assert pipeline.get_field("sec.unit_kg") is True

    def test_every_categorical_in_a_section_is_encoded(self):
        sections = {
            "base": {
                "scale": field("scale", "categorical"),
                "unit": field("unit", "categorical"),
            }
        }
        pipeline = make_pipeline({"scale": "celsius", "unit": "kg"}, sections)
        assert pipeline.field_dict["scale_celsius"] is True
        assert pipeline.field_dict["unit_kg"] is True

    def test_non_categorical_types_are_not_one_hot_encoded(self):
        pipeline = make_pipeline({"x": "abc"}, {"base": {"x": field("x", "string")}})
        assert pipeline.field_dict == {"x": "abc"}

    @pytest.mark.parametrize(
        "supplied,expected_one_hot",
        [(3, "quarter_3"), (0, "quarter_0"), (True, "quarter_True")],
    )
    def test_integer_categorical_value_is_one_hot_encoded(self, supplied, expected_one_hot):
        """Integer categories are legitimate schema design — a hurricane category 1-5,
        a quarter, a star rating — and an LLM emits them as JSON numbers, not strings.

        Regression: the one-hot name was built with ``field + '_' + value``, which
        raised a raw TypeError for any non-string value.
        """
        sections = {"base": {"quarter": field("quarter", "categorical")}}
        pipeline = make_pipeline({"quarter": supplied}, sections)
        assert pipeline.field_dict[expected_one_hot] is True

    def test_integer_and_string_categorical_agree(self):
        """A schema must not depend on whether the LLM typed the value as a JSON
        number or a JSON string."""
        sections = {"base": {"quarter": field("quarter", "categorical")}}
        as_number = make_pipeline({"quarter": 3}, sections).field_dict
        as_string = make_pipeline({"quarter": "3"}, sections).field_dict
        assert "quarter_3" in as_number and "quarter_3" in as_string

    @pytest.mark.parametrize("supplied", [2.5, 2.0, 0.0, -1.5])
    def test_float_categorical_value_is_rejected(self, supplied):
        """Floats are not a valid categorical value: ``str(2.5)`` contains a ``.``,
        which is dsl-spa's field-path separator, so ``set_field`` would split the
        one-hot name and bury it as a nested dict (``{'quarter_2': {'5': True}}``)
        instead of a flat key. There is no use case for a float category, so this is
        rejected outright rather than escaped.

        ``2.0`` matters as much as ``2.5`` — JSON ``3.0`` parses to a Python float, so
        an LLM writing a whole number with a decimal point hits this too.
        """
        sections = {"base": {"quarter": field("quarter", "categorical")}}
        with pytest.raises(PipelineException):
            make_pipeline({"quarter": supplied}, sections)

    def test_float_categorical_rejection_names_the_field(self):
        sections = {"sec": {"quarter": field("quarter", "categorical")}}
        with pytest.raises(PipelineException) as excinfo:
            make_pipeline({"sec": {"quarter": 2.5}}, sections)
        assert "sec.quarter" in str(excinfo.value)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-8: fill_categorical_values builds the path from the dict key "
        "while defaults and required fields use d['name'] - they disagree",
    )
    def test_categorical_resolves_by_field_name_like_defaults_do(self):
        sections = {"base": {"keyname": field("realname", "categorical")}}
        pipeline = make_pipeline({"realname": "x"}, sections)
        assert pipeline.field_dict["realname_x"] is True

    def test_two_level_section_categorical_is_encoded(self):
        """Regression: 'root + key' concatenated a list with a str -> TypeError."""
        sections = {"outer": {"inner": {"unit": field("unit", "categorical")}}}
        pipeline = make_pipeline({"outer": {"inner": {"unit": "kg"}}}, sections)
        assert pipeline.field_dict["outer"]["inner"]["unit_kg"] is True


class TestRequiredFields:
    def test_single_required_field_is_listed_with_its_section_path(self):
        pipeline = make_pipeline({"a": "1"}, {"base": {"a": field("a", required=True)}})
        assert pipeline.required_fields == ["base.a"]

    def test_present_required_field_constructs_cleanly(self):
        pipeline = make_pipeline({"a": "1"}, {"base": {"a": field("a", required=True)}})
        assert pipeline.get_field("a") == "1"

    def test_missing_required_field_raises(self):
        with pytest.raises(PipelineException):
            make_pipeline({}, {"base": {"a": field("a", required=True)}})

    def test_missing_required_field_message_names_the_field(self):
        with pytest.raises(PipelineException) as excinfo:
            make_pipeline({}, {"base": {"a": field("a", required=True)}})
        assert "base.a" in str(excinfo.value)

    def test_all_optional_section_has_no_required_fields(self):
        sections = {"base": {"a": field("a"), "b": field("b")}}
        assert make_pipeline({}, sections).required_fields == []

    def test_required_field_supplied_as_null_is_rejected(self):
        """Null is not a valid field value, so supplying null for a required field is
        rejected at construction just as omitting it would be."""
        sections = {"base": {"a": field("a", required=True)}}
        with pytest.raises(PipelineException):
            make_pipeline({"a": None}, sections)

    def test_required_field_satisfied_by_its_default(self):
        sections = {"base": {"a": field("a", required=True, default="dv")}}
        assert make_pipeline({}, sections).get_field("a") == "dv"

    def test_required_fields_in_separate_sections_are_all_detected(self):
        sections = {
            "s1": {"a": field("a", required=True)},
            "s2": {"b": field("b", required=True)},
        }
        pipeline = make_pipeline({"s1": {"a": "1"}, "s2": {"b": "2"}}, sections)
        assert pipeline.required_fields == ["s1.a", "s2.b"]

    def test_check_for_required_fields_accepts_a_satisfied_list(self):
        pipeline = make_pipeline({"a": "1"}, {"base": {"a": field("a")}})
        pipeline.check_for_required_fields(["a"])

    def test_check_for_required_fields_rejects_a_missing_field(self):
        pipeline = make_pipeline({}, {})
        with pytest.raises(PipelineException):
            pipeline.check_for_required_fields(["nope"])

    def test_second_required_field_in_a_section_is_enforced(self):
        """Regression: build_required_fields_list used to return from inside its
        per-key loop, so only the first field of a section was ever inspected."""
        sections = {
            "base": {
                "a": field("a", required=True),
                "b": field("b", required=True),
            }
        }
        with pytest.raises(PipelineException):
            make_pipeline({"a": "1"}, sections)

    def test_required_field_after_an_optional_one_is_detected(self):
        """Regression: an optional first field used to short-circuit the loop with
        'return []', so the section reported zero required fields."""
        sections = {
            "base": {
                "a": field("a", required=False),
                "b": field("b", required=True),
            }
        }
        pipeline = make_pipeline({"a": "1", "b": "2"}, sections)
        assert "base.b" in pipeline.required_fields

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-9: a field declared outside any section builds the path "
        "f'{root}.{name}' with root == '', yielding a leading dot ('.x')",
    )
    def test_field_declared_outside_a_section_gets_a_clean_path(self):
        pipeline = make_pipeline({"x": "1"}, {"x": field("x", required=True)})
        assert pipeline.required_fields == ["x"]

    def test_two_level_section_required_field_is_detected(self):
        """Regression: a misplaced paren passed 'root' to list.extend() instead of to
        the recursive call, raising TypeError on any tree nested past one level."""
        pipeline = make_pipeline({}, {})
        sections = {"outer": {"inner": {"a": field("a", required=True)}}}
        assert pipeline.build_required_fields_list(sections) == ["outer.inner.a"]


class TestCheckForField:
    def test_present_field(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("s") is True

    def test_absent_field(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("nope") is False

    def test_present_nested_field(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("sec.n") is True

    def test_absent_nested_field(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("sec.nope") is False

    def test_absent_section(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("nosuch.n") is False

    def test_base_prefix_is_stripped(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("base.s") is True

    def test_base_segment_is_skipped_anywhere_in_the_path(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("sec.base.n") is True

    def test_field_holding_false_counts_as_present(self, accessor_pipeline):
        """Categorical one-hots are booleans, so False must not read as absent."""
        assert accessor_pipeline.check_for_field("flag") is True

    def test_section_itself_counts_as_present(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("sec") is True

    def test_field_holding_none_is_treated_as_absent(self, accessor_pipeline):
        """By design, a null field is not a valid value - null is rejected, so it
        reads as absent. Contrast with False, which is a legitimate value."""
        assert accessor_pipeline.check_for_field("nothing") is False

    def test_nested_field_holding_none_is_treated_as_absent(self):
        pipeline = make_pipeline({"sec": {"v": None}}, {"sec": {"v": field("v")}})
        assert pipeline.check_for_field("sec.v") is False

    def test_null_field_is_reported_as_missing(self, accessor_pipeline):
        assert accessor_pipeline.get_list_of_missing_fields(["nothing"]) == ["nothing"]

    @pytest.mark.xfail(
        strict=True,
        reason="out of scope (BUG-7): 'base' is skipped as a path segment, so the "
        "bare string 'base' resolves to the whole root dict and reports as present. "
        "Requires a schema using 'base' alone where a field name belongs - not "
        "expressible via PipelineField and not something anyone writes",
    )
    def test_bare_base_is_not_a_field(self, accessor_pipeline):
        assert accessor_pipeline.check_for_field("base") is False


class TestGetField:
    def test_present_field(self, accessor_pipeline):
        assert accessor_pipeline.get_field("s") == "v"

    def test_base_prefix_is_stripped(self, accessor_pipeline):
        assert accessor_pipeline.get_field("base.s") == "v"

    def test_nested_field(self, accessor_pipeline):
        assert accessor_pipeline.get_field("sec.n") == 1

    def test_section_is_returned_whole(self, accessor_pipeline):
        assert accessor_pipeline.get_field("sec") == {"n": 1}

    def test_field_holding_false_is_returned_as_false(self, accessor_pipeline):
        """The counterpart to the test below: a real False must survive as False."""
        assert accessor_pipeline.get_field("flag") is False

    def test_missing_field_raises_pipeline_exception(self, accessor_pipeline):
        """Regression: used to return False for a missing field, which was
        indistinguishable from a field legitimately holding False (every categorical
        one-hot is a bool)."""
        with pytest.raises(PipelineException):
            accessor_pipeline.get_field("definitely_missing")

    def test_missing_field_message_names_the_field(self, accessor_pipeline):
        with pytest.raises(PipelineException) as excinfo:
            accessor_pipeline.get_field("definitely_missing")
        assert "definitely_missing" in str(excinfo.value)

    def test_missing_nested_field_raises(self, accessor_pipeline):
        with pytest.raises(PipelineException):
            accessor_pipeline.get_field("sec.nope")

    def test_field_holding_none_raises_like_a_missing_field(self, accessor_pipeline):
        """Null is rejected by design, so reading one is an error, not a None return."""
        with pytest.raises(PipelineException):
            accessor_pipeline.get_field("nothing")

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-11: descending past a scalar calls .keys() on it, raising "
        "AttributeError instead of PipelineException",
    )
    def test_descending_through_a_scalar_raises_pipeline_exception(self, accessor_pipeline):
        with pytest.raises(PipelineException):
            accessor_pipeline.get_field("s.deeper")


class TestSetField:
    def test_overwrites_an_existing_value(self, accessor_pipeline):
        accessor_pipeline.set_field("s", "new")
        assert accessor_pipeline.get_field("s") == "new"

    def test_creates_a_new_root_level_field(self, accessor_pipeline):
        accessor_pipeline.set_field("fresh", 42)
        assert accessor_pipeline.field_dict["fresh"] == 42

    def test_writes_into_an_existing_section(self, accessor_pipeline):
        accessor_pipeline.set_field("sec.m", 2)
        assert accessor_pipeline.field_dict["sec"] == {"n": 1, "m": 2}

    def test_creates_intermediate_sections_as_needed(self, accessor_pipeline):
        accessor_pipeline.set_field("a.b.c", 3)
        assert accessor_pipeline.field_dict["a"] == {"b": {"c": 3}}

    def test_base_prefix_is_stripped(self, accessor_pipeline):
        accessor_pipeline.set_field("base.fresh", "nv")
        assert accessor_pipeline.field_dict["fresh"] == "nv"

    def test_round_trips_through_get_field(self, accessor_pipeline):
        accessor_pipeline.set_field("deep.path.value", "x")
        assert accessor_pipeline.get_field("deep.path.value") == "x"
        assert accessor_pipeline.check_for_field("deep.path.value") is True

    def test_can_store_a_false_value(self, accessor_pipeline):
        accessor_pipeline.set_field("off", False)
        assert accessor_pipeline.get_field("off") is False


class TestGetListOfMissingFields:
    def test_all_present_returns_empty(self, accessor_pipeline):
        assert accessor_pipeline.get_list_of_missing_fields(["s", "sec.n"]) == []

    def test_returns_only_the_missing_fields_in_order(self, accessor_pipeline):
        result = accessor_pipeline.get_list_of_missing_fields(["s", "gone", "sec.n", "also"])
        assert result == ["gone", "also"]

    def test_empty_input_returns_empty(self, accessor_pipeline):
        assert accessor_pipeline.get_list_of_missing_fields([]) == []


class TestAddFieldsToClause:
    def test_single_placeholder(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("val={s}") == "val=v"

    def test_two_distinct_placeholders(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("{s} and {sec.n}") == "v and 1"

    def test_dotted_section_path(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("n={sec.n}") == "n=1"

    def test_base_prefixed_placeholder(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("{base.s}") == "v"

    def test_value_longer_than_its_placeholder(self):
        pipeline = make_pipeline({"s": "LONGVALUE"}, {"base": {"s": field("s")}})
        assert pipeline.add_fields_to_clause("{s}{s}") == "LONGVALUELONGVALUE"

    def test_missing_field_is_left_verbatim(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("x={gone} end") == "x={gone} end"

    def test_missing_placeholder_does_not_block_a_later_one(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("{gone} {s}") == "{gone} v"

    def test_clause_without_braces_is_returned_unchanged(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("base.s") == "base.s"

    def test_empty_clause(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("") == ""

    def test_unclosed_brace_terminates(self, accessor_pipeline):
        """Loop-termination guard: no '}' must not hang or raise."""
        assert accessor_pipeline.add_fields_to_clause("hello {s") == "hello {s"

    def test_stray_closing_brace_before_a_placeholder(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("a } b {s}") == "a } b v"

    def test_substituted_value_containing_braces_is_not_re_expanded(self):
        """Loop-termination guard: a self-referential value must not loop forever."""
        pipeline = make_pipeline({"s": "{s}"}, {"base": {"s": field("s")}})
        assert pipeline.add_fields_to_clause("{s}") == "{s}"

    @pytest.mark.parametrize(
        "value,expected",
        [(5, "5"), (1.5, "1.5"), (True, "True"), (0, "0"), ("", "")],
    )
    def test_non_string_values_are_stringified(self, value, expected):
        pipeline = make_pipeline({"v": value}, {"base": {"v": field("v")}})
        assert pipeline.add_fields_to_clause("[{v}]") == f"[{expected}]"

    def test_none_value_placeholder_is_left_verbatim_like_a_missing_field(self):
        """Null is rejected by design, so a null field behaves as absent here - the
        placeholder is left alone, exactly as for a field that was never supplied."""
        pipeline = make_pipeline({"v": None}, {"base": {"v": field("v")}})
        assert pipeline.add_fields_to_clause("[{v}]") == "[{v}]"

    def test_null_field_does_not_block_a_later_placeholder(self):
        sections = {"base": {"v": field("v"), "w": field("w")}}
        pipeline = make_pipeline({"v": None, "w": "x"}, sections)
        assert pipeline.add_fields_to_clause("{v}{w}") == "{v}x"

    def test_sanitize_for_sql_escapes_single_quotes(self):
        pipeline = make_pipeline({"s": "O'Brien"}, {"base": {"s": field("s")}})
        result = pipeline.add_fields_to_clause("name='{s}'", sanitize_for_sql=True)
        assert result == "name='O\\'Brien'"

    def test_quotes_are_left_alone_without_sanitize_for_sql(self):
        pipeline = make_pipeline({"s": "O'Brien"}, {"base": {"s": field("s")}})
        assert pipeline.add_fields_to_clause("name='{s}'") == "name='O'Brien'"

    def test_sanitize_for_sql_ignores_non_string_values(self, accessor_pipeline):
        assert accessor_pipeline.add_fields_to_clause("{n}", sanitize_for_sql=True) == "5"

    def test_sanitize_for_sql_applies_to_every_placeholder(self):
        pipeline = make_pipeline(
            {"a": "O'A", "b": "O'B"},
            {"base": {"a": field("a"), "b": field("b")}},
        )
        result = pipeline.add_fields_to_clause("{a}|{b}", sanitize_for_sql=True)
        assert result == "O\\'A|O\\'B"

    def test_repeated_placeholder_is_substituted_every_time(self, accessor_pipeline):
        """Regression: the cursor used to apply a pre-substitution offset to the
        post-substitution string, so '{s}-{s}-{s}' yielded 'v-{s}-v'."""
        assert accessor_pipeline.add_fields_to_clause("{s}-{s}-{s}") == "v-v-v"

    def test_empty_value_does_not_swallow_the_next_placeholder(self):
        """Regression: an empty-string substitution shifted the cursor past the
        following placeholder, yielding '[][{s}]'."""
        pipeline = make_pipeline({"s": ""}, {"base": {"s": field("s")}})
        assert pipeline.add_fields_to_clause("[{s}][{s}]") == "[][]"

    def test_short_value_does_not_skip_a_following_distinct_placeholder(self):
        """Regression: with a short value the cursor overshot, so the trailing
        placeholder was never seen."""
        pipeline = make_pipeline(
            {"a": "", "b": "B"},
            {"base": {"a": field("a"), "b": field("b")}},
        )
        assert pipeline.add_fields_to_clause("{a}{b}") == "B"


class TestSanitizeFieldForSqlQuery:
    def test_single_quote_is_escaped(self, accessor_pipeline):
        assert accessor_pipeline.sanitize_field_for_sql_query("O'Brien") == "O\\'Brien"

    def test_every_quote_is_escaped(self, accessor_pipeline):
        assert accessor_pipeline.sanitize_field_for_sql_query("a'b'c") == "a\\'b\\'c"

    def test_string_without_quotes_is_unchanged(self, accessor_pipeline):
        assert accessor_pipeline.sanitize_field_for_sql_query("plain") == "plain"

    def test_empty_string(self, accessor_pipeline):
        assert accessor_pipeline.sanitize_field_for_sql_query("") == ""

    def test_already_escaped_input_is_escaped_again(self, accessor_pipeline):
        """Documents that sanitizing twice double-escapes; callers must not."""
        assert accessor_pipeline.sanitize_field_for_sql_query("\\'") == "\\\\'"

    @pytest.mark.xfail(
        strict=True,
        reason="out of scope (BUG-13): typed (Any) -> Any but calls .replace, so a "
        "non-string raises AttributeError. Unreachable - the sole call site guards with "
        "isinstance(value, str) and nothing else calls it",
    )
    def test_non_string_input_is_returned_unchanged(self, accessor_pipeline):
        assert accessor_pipeline.sanitize_field_for_sql_query(5) == 5


class TestValidateFieldTypes:
    """Type coercion. The docs promise validation for str, int and float types.

    ``validate_field_types`` only works when handed a single section's dict; see
    BUG-3 for why the documented behavior never actually happens in practice.
    """

    @pytest.mark.parametrize(
        "field_type,supplied,expected",
        [
            ("string", 9, "9"),
            ("str", 9, "9"),
            ("integer", "5", 5),
            ("int", "5", 5),
            ("float", "1.5", 1.5),
            ("number", "1.5", 1.5),
        ],
    )
    def test_coerces_a_section_dict(self, field_type, supplied, expected):
        section = {"v": field("v", field_type)}
        pipeline = make_pipeline({"v": supplied}, {"base": section})
        pipeline.validate_field_types(section)
        assert pipeline.get_field("v") == expected
        assert isinstance(pipeline.get_field("v"), type(expected))

    def test_already_correct_values_are_left_alone(self):
        section = {"s": field("s", "string"), "i": field("i", "integer")}
        pipeline = make_pipeline({"s": "text", "i": 7}, {"base": section})
        pipeline.validate_field_types(section)
        assert pipeline.field_dict == {"s": "text", "i": 7}

    def test_int_value_is_widened_for_a_float_field(self):
        section = {"f": field("f", "number")}
        pipeline = make_pipeline({"f": 5}, {"base": section})
        pipeline.validate_field_types(section)
        assert isinstance(pipeline.get_field("f"), float)
        assert pipeline.get_field("f") == 5.0

    def test_unconvertible_value_raises_pipeline_exception(self):
        """Now raised at construction, since update_fields runs the coercion."""
        sections = {"base": {"i": field("i", "integer")}}
        with pytest.raises(PipelineException):
            make_pipeline({"i": "not a number"}, sections)

    def test_unconvertible_value_message_names_the_field(self):
        sections = {"base": {"i": field("i", "integer")}}
        with pytest.raises(PipelineException) as excinfo:
            make_pipeline({"i": "not a number"}, sections)
        assert "base.i" in str(excinfo.value)

    def test_unconvertible_value_raises_on_a_direct_call_too(self):
        section = {"i": field("i", "integer")}
        pipeline = make_pipeline({}, {"base": section})
        pipeline.set_field("i", "not a number")
        with pytest.raises(PipelineException):
            pipeline.validate_field_types(section)

    def test_absent_fields_are_skipped(self):
        section = {"i": field("i", "integer")}
        pipeline = make_pipeline({}, {"base": section})
        pipeline.validate_field_types(section)
        assert pipeline.field_dict == {}

    @pytest.mark.parametrize("field_type", ["datetime", "categorical", "bool"])
    def test_untyped_and_unknown_types_are_left_untouched(self, field_type):
        section = {"v": field("v", field_type)}
        pipeline = make_pipeline({"v": "untouched"}, {"base": section})
        pipeline.validate_field_types(section)
        assert pipeline.get_field("v") == "untouched"

    @pytest.mark.parametrize(
        "field_type,supplied,expected",
        [("string", 9, "9"), ("integer", "5", 5), ("float", "1.5", 1.5)],
    )
    def test_coerces_a_full_fields_schema(self, field_type, supplied, expected):
        """Regression: the recursion used to delegate to fill_categorical_values
        instead of itself, so passing the full 'fields' tree coerced nothing."""
        sections = {"base": {"v": field("v", field_type)}}
        pipeline = make_pipeline({"v": supplied}, sections)
        pipeline.validate_field_types(sections)
        assert pipeline.get_field("v") == expected

    def test_construction_coerces_declared_types(self):
        """Regression: validate_field_types was never called from update_fields, so
        the type validation the docs promise never actually ran."""
        sections = {
            "base": {
                "i": field("i", "integer"),
                "f": field("f", "float"),
                "s": field("s", "string"),
            }
        }
        pipeline = make_pipeline({"i": "5", "f": "1.5", "s": 9}, sections)
        assert pipeline.field_dict == {"i": 5, "f": 1.5, "s": "9"}

    @pytest.mark.xfail(
        strict=True,
        reason="out of scope (BUG-15): isinstance(True, int) is True, so a bool passes "
        "as an 'integer' field unchallenged. Requires the LLM to contradict the field's "
        "declared type, which is a prompt defect rather than a framework one. Contrast "
        "BUG-12, where the LLM is correct and the framework cannot accept valid input",
    )
    def test_bool_is_not_accepted_as_an_integer(self):
        section = {"i": field("i", "integer")}
        pipeline = make_pipeline({"i": True}, {"base": section})
        with pytest.raises(PipelineException):
            pipeline.validate_field_types(section)


class TestUpdateFields:
    CATEGORICAL = {"base": {"c": field("c", "categorical")}}

    def test_replaces_the_field_dict(self):
        pipeline = make_pipeline({"c": "red"}, self.CATEGORICAL)
        pipeline.update_fields({"c": "blue"})
        assert pipeline.field_dict["c"] == "blue"

    def test_re_encodes_the_new_categorical_value(self):
        pipeline = make_pipeline({"c": "red"}, self.CATEGORICAL)
        pipeline.update_fields({"c": "blue"})
        assert pipeline.field_dict["c_blue"] is True

    def test_drops_the_stale_one_hot_from_the_previous_value(self):
        pipeline = make_pipeline({"c": "red"}, self.CATEGORICAL)
        pipeline.update_fields({"c": "blue"})
        assert "c_red" not in pipeline.field_dict

    def test_is_idempotent_when_fed_its_own_field_dict(self):
        pipeline = make_pipeline({"c": "red"}, self.CATEGORICAL)
        before = copy.deepcopy(pipeline.field_dict)
        pipeline.update_fields(pipeline.field_dict)
        assert pipeline.field_dict == before

    def test_reapplies_defaults(self):
        sections = {"base": {"x": field("x", default="dv")}}
        pipeline = make_pipeline({"x": "given"}, sections)
        pipeline.update_fields({})
        assert pipeline.field_dict == {"x": "dv"}

    def test_recomputes_required_fields(self):
        sections = {"base": {"a": field("a", required=True)}}
        pipeline = make_pipeline({"a": "1"}, sections)
        pipeline.update_fields({"a": "2"})
        assert pipeline.required_fields == ["base.a"]

    def test_dropping_a_required_field_raises(self):
        sections = {"base": {"a": field("a", required=True)}}
        pipeline = make_pipeline({"a": "1"}, sections)
        with pytest.raises(PipelineException):
            pipeline.update_fields({})
