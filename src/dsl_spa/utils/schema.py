from typing import Any, Union

class PipelineSchemaException(Exception):
    """Pipeline Schema Exception. A PipelineSchemaException is created when an error is generated trying to create the Pipeline Schema.
    Distinguishes exceptions generated based on the pipeline schema from exceptions generated from standard python errors.
    """
    def __init__(self, error: str):
        """Creates PipelineException

        Args:
            error (str): Pipeline Error
        """
        self.pipeline_error = error
        super().__init__(error)
        
    def __str__(self):
        return f"Pipeline Schema Exception: {self.pipeline_error}"


class PipelineComponent:
    """Base PipelineComponent class.
    """        
    def generate_schema(self) -> dict:
        """Generates Schema for Pipeline Component

        Returns:
            dict: Dictionary defining Pipeline Component
        """
        raise NotImplementedError("Use one of the subclasses for your specific need")

class PipelineField(PipelineComponent):
    """Pipeline Field Definition
    """
    def __init__(self, field_name: str, field_type: str, required: bool, description: str, section_name: str = "base", default: Any = None):
        """Definition for Field in Pipeline.

        Args:
            field_name (str): Field Name in schema
            field_type (str): Type of Field
            required (bool): Whether Field is Required for Pipeline
            description (str): Description of Field
            section_name (str, optional): Section for Field. If schema is flat use "base". Defaults to "base".
            default (Any, optional): default value for Field. Defaults to "base".
        """
        if not isinstance(field_name, str):
            raise PipelineSchemaException(f"field_name should be a str, found {type(field_name)}")
        if not isinstance(field_type, str):
            raise PipelineSchemaException(f"field_type should be a str, found {type(field_type)}")
        if not isinstance(required, bool):
            raise PipelineSchemaException(f"field_name required be a bool, found {type(required)}")
        if not isinstance(description, str):
            raise PipelineSchemaException(f"description should be a str, found {type(description)}")
        if not isinstance(section_name, str):
            raise PipelineSchemaException(f"section_name should be a str, found {type(section_name)}")
        self.name = field_name
        self.type = field_type
        self.required = required
        self.description = description
        self.section = section_name
        self.default = default
        
    def generate_schema(self):
        """Generates Schema for Pipeline Field

        Returns:
            dict: Dictionary defining Pipeline Field
        """
        field_schema = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description
        }
        if self.default is not None:
            field_schema["default"] = self.default
        return field_schema
        
    def get_name(self) -> str:
        """Gets Field Name

        Returns:
            str: Field Name
        """
        return self.name
        
    def get_section(self) -> str:
        """Gets Field Section

        Returns:
            str: Field Section
        """
        return self.section

class Query(PipelineComponent):
    """Query Definition
    """
    def __init__(self, query_name: str, connector_name: str):
        """Creates Query Definition

        Args:
            query_name (str): Name of Query
            connector_name (str): Name of Connector
        """
        if not isinstance(query_name, str):
            raise PipelineSchemaException(f"query_name should be a str, found {type(query_name)}")
        if not isinstance(connector_name, str):
            raise PipelineSchemaException(f"connector_name should be a str, found {type(connector_name)}")
        self.name = query_name
        self.connector = connector_name
        self.clauses = []
    
    def get_name(self) -> str:
        """Gets Query Name

        Returns:
            str: Query Name
        """
        return self.name
        
    def add_clause(self, clause: str, optional: bool, field_required: str = None):
        """Adds Clause to Query definition

        Args:
            clause (str): database query clause to add
            optional (bool): whether clause is optional
            field_required (str, optional): The required field to run. Set this to none unless optional is true. Defaults to None.
        """
        if not isinstance(clause, str):
            raise PipelineSchemaException(f"clause should be a str, found {type(clause)}")
        if not isinstance(optional, str):
            raise PipelineSchemaException(f"optional should be a bool, found {type(optional)}")
        if field_required is not None and not isinstance(field_required, str):
            raise PipelineSchemaException(f"field_required should be a str, found {type(field_required)}")
        clause = {
            "clause": clause,
            "optional": optional
        }
        if optional:
            clause["field"] = field_required
        self.clauses.append(clause)
        
    def generate_schema(self) -> dict:
        """Generates Schema for Pipeline Query

        Returns:
            dict: Dictionary defining Pipeline Query
        """
        query_dict = {
            "name": self.name,
            "connector": self.connector,
            "clauses": self.clauses
        }
        return query_dict
        
class AdvancedQuery(Query):
    """Advanced Query Definition
    """
    def __init__(self, query_name: str, connector_name: str, min_results: int = None, error_message: str = None, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Creates an Advanced Query Definition

        Args:
            query_name (str): Name of Query
            connector_name (str): Name of Connector
            min_results (int, optional): Minimum results query must have to not cause an error. If None is excluded from schema. Defaults to None.
            error_message (str, optional): Error message to display if minimum results requirement is not met. If None is excluded. Defaults to None.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        super().__init__(query_name,connector_name)
        if min_results is not None and not isinstance(min_results, int):
            raise PipelineSchemaException(f"min_results should be a int, found {type(min_results)}")
        if error_message is not None and not isinstance(error_message, str):
            raise PipelineSchemaException(f"error_message should be a str, found {type(error_message)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        self.min_results = min_results
        self.error_message = error_message
        self.required_fields = required_fields
        self.exclude_fields = exclude_fields
        
    def generate_schema(self) -> dict:        
        """Generates Schema for Pipeline Advanced Query

        Returns:
            dict: Dictionary defining Pipeline Advanced Query
        """
        query_dict = super().generate_schema()
        if self.min_results is not None:
            query_dict["min_results"] = self.min_results
        if self.error_message is not None:
            query_dict["error"] = self.error_message
        if self.required_fields is not None:
            query_dict["required_fields"] = self.required_fields
        if self.exclude_fields is not None:
            query_dict["exclude_fields"] = self.exclude_fields
        return query_dict
    
class CSV(PipelineComponent):
    """CSV Definition
    """
    def __init__(self, csv_name: str, connector_name: str, filename: str):
        """Creates CSV Definition

        Args:
            csv_name (str): Name of CSV to be used in other pipeline definitions
            connector_name (str): Name of Connector
            filename (str): Name of csv file in local storage
        """
        if not isinstance(csv_name, str):
            raise PipelineSchemaException(f"csv_name should be a str, found {type(csv_name)}")
        if not isinstance(connector_name, str):
            raise PipelineSchemaException(f"connector_name should be a str, found {type(connector_name)}")
        if not isinstance(filename, str):
            raise PipelineSchemaException(f"filename should be a str, found {type(filename)}")
        self.name = csv_name
        self.connector = connector_name
        self.csv_name = filename
        self.column_filters = []
    
    def add_column_filter(self, field_name: str, column_name: str, value: Any):
        """Adds a column filter to the csv definition

        Args:
            field_name (str): Field Name required for column filter to be applied
            column_name (str): Name of column to filter the CSV on
            value (Any): Value to filter column on (can be field value by using '{field name}')
        """
        if not isinstance(field_name, str):
            raise PipelineSchemaException(f"field_name should be a str, found {type(field_name)}")
        if not isinstance(column_name, str):
            raise PipelineSchemaException(f"column_name should be a str, found {type(column_name)}")
        self.column_filters.append({
            "field": field_name,
            "column": column_name,
            "value": value
        })
        
    def generate_schema(self):
        csv_dict = {
            "name": self.name,
            "connector": self.connector,
            "csv_name": self.csv_name
        }
        if len(self.column_filters) > 0:
            csv_dict["column_filters"] = []
            for column_filter in self.column_filters:
                csv_dict["column_filters"].append({
                    "field": column_filter["field"],
                    "column": column_filter["column"],
                    "value": column_filter["value"]
                })
        return csv_dict
    
class Filter(PipelineComponent):
    """Filter Definition
    """
    def __init__(self, filter_name: str, display_name: str, column_name: str, query_name: str, include_any: bool = True):
        """Creates Filter Definition

        Args:
            filter_name (str): Name of Filter
            display_name (str): Display Name for Filter
            column_name (str): Column Name of Filter
            query_name (str): Query Name to load filter values from
            include_any (bool, optional): Whether to include "Any" as an option in the filter. Defaults to True.
        """
        if not isinstance(filter_name, str):
            raise PipelineSchemaException(f"filter_name should be a str, found {type(filter_name)}")
        if not isinstance(display_name, str):
            raise PipelineSchemaException(f"display_name should be a str, found {type(display_name)}")
        if not isinstance(column_name, str):
            raise PipelineSchemaException(f"column_name should be a str, found {type(column_name)}")
        if not isinstance(query_name, str):
            raise PipelineSchemaException(f"query_name should be a str, found {type(query_name)}")
        if include_any is not None and not isinstance(include_any, bool):
            raise PipelineSchemaException(f"include_any should be a str, found {type(include_any)}")
        self.name = filter_name
        self.display_name = display_name
        self.column_name = column_name
        self.query = query_name
        self.include_any = include_any
        
    def get_name(self) -> str:
        """Gets Filter Name

        Returns:
            str: Filter Name
        """
        return self.name
    
    def generate_schema(self):
        """Generates Schema for Pipeline Filter

        Returns:
            dict: Dictionary defining Pipeline Filter
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "column_name": self.column_name,
            "include_any": self.include_any,
            "query": self.query
        }    
        
class Dataset(PipelineComponent):
    """Dataset Definition
    """
    def __init__(self, dataset_name: str):
        """Creates Dataset Definition

        Args:
            dataset_name (str): Name of Dataset
        """
        if not isinstance(dataset_name, str):
            raise PipelineSchemaException(f"dataset_name should be a str, found {type(dataset_name)}")
        self.name = dataset_name
        self.create_schema = None
        self.operations = []
        
    def get_name(self) -> str:
        """Gets Dataset Name

        Returns:
            str: Dataset Name
        """
        return self.name
        
    def create_from_query(self, query_name: str):
        """Adds Create from Query operation to dataset

        Args:
            query_name (str): Name of Query

        Raises:
            ValueError: Adding too many create operations
        """
        if not isinstance(query_name, str):
            raise PipelineSchemaException(f"query_name should be a str, found {type(query_name)}")
        if self.create_schema is not None:
            raise ValueError("A dataset can only have one create operation.")
            
        self.create_schema = {
            "type": "query",
            "name": query_name
        }
    
    def create_from_dataset(self, dataset_name: str):
        """Adds Create from Query operation to dataset

        Args:
            dataset_name (str): Name o Dataset

        Raises:
            ValueError: Adding too many create operations
        """
        if not isinstance(dataset_name, str):
            raise PipelineSchemaException(f"dataset_name should be a str, found {type(dataset_name)}")
        if self.create_schema is not None:
            raise ValueError("A dataset can only have one create operation.")
        
        self.create_schema = {
            "type": "dataset",
            "name": dataset_name
        }
        
    def merge_two_datasets(self, dataset1_name: str, dataset2_name: str, how: str, left_on: str, right_on: str, nan_replace: Any = None):
        """Adds Create from Merge operation to dataset

        Args:
            dataset1_name (str): Name of Left Dataset
            dataset2_name (str): Name of Right Dataset
            how (str): How to merge [left,right,inner,outer].
            left_on (str): Column in dataset 1 to use for merge
            right_on (str): Column in dataset 2 to use for merge
            nan_replace (Any, optional): Value to replace nan values with. If None does not replace nan values. Defaults to None.

        Raises:
            ValueError: Adding too many create operations
            AttributeError: How must be one of [left,right,inner,outer]
        """
        if not isinstance(dataset1_name, str):
            raise PipelineSchemaException(f"dataset1_name should be a str, found {type(dataset1_name)}")
        if not isinstance(dataset2_name, str):
            raise PipelineSchemaException(f"dataset2_name should be a str, found {type(dataset2_name)}")
        if not isinstance(how, str):
            raise PipelineSchemaException(f"how should be a str, found {type(how)}")
        if not isinstance(left_on, str):
            raise PipelineSchemaException(f"left_on should be a str, found {type(left_on)}")
        if not isinstance(right_on, str):
            raise PipelineSchemaException(f"right_on should be a str, found {type(right_on)}")
        if self.create_schema is not None:
            raise PipelineSchemaException("A dataset can only have one create operation (create from query/dataset or merge two datasets).")
        
        how = how.lower()
        if how not in ["left","right","inner","outer"]:
            raise AttributeError('how must be one of "left","right","inner","outer"')
        self.create_schema = {
            "type": "merge",
            "dataset1": dataset1_name,
            "dataset2": dataset2_name,
            "how": how,
            "left_on": left_on,
            "right_on": right_on
        }
        if nan_replace is not None:
            self.create_schema["nan_replace"] = nan_replace
        
    def add_function(self, function_name: str, function_fields_dict: dict = None, function_params_dict: dict = None):
        """Adds function to Dataset Operations

        Args:
            function_name (str): Name of function in Pipelines functions dict
            function_fields_dict (dict, optional): Dictionary of Fields to map to parameters for the function. If None is excluded from schema. Defaults to None.
            function_params_dict (dict, optional): Dictionary of static parameters to map to parameters for the function. If None is excluded from schema. Defaults to None.
        """
        if not isinstance(function_name, str):
            raise PipelineSchemaException(f"function_name should be a str, found {type(function_name)}")
        if function_fields_dict is not None and not isinstance(function_fields_dict, dict):
            raise PipelineSchemaException(f"function_fields_dict should be a dict, found {type(function_fields_dict)}")
        if function_params_dict is not None and not isinstance(function_params_dict, dict):
            raise PipelineSchemaException(f"function_params_dict should be a dict, found {type(function_params_dict)}")
        
        function_dict = {
            "type" : "function",
            "name" : function_name
        }
            
        if function_fields_dict is not None:
            function_dict["fields"] = function_fields_dict
            
        if function_params_dict is not None:
            function_dict["params"] = function_params_dict
            
        self.operations.append(function_dict)
    
    def add_filter(self, filters_list: list[str]):
        """Adds filter to Dataset Operations

        Args:
            filters_list (list[str]): List of Filters to apply
        """
        if not isinstance(filters_list, list):
            raise PipelineSchemaException(f"filters_list should be a list, found {type(filters_list)}")
        self.operations.append({
            "type": "filter",
            "filters": filters_list
        })
        
    def add_arithmetic_operation(self, arithmetic_operator: str, column: str, by: float):
        """Adds Arithmetic Operation to Dataset Operations

        Args:
            arithmetic_operator (str): Arithmetic Operator
            column (str): Name of Column to apply operator to
            by (float): value to apply operation by

        Raises:
            AttributeError: _description_
        """
        if not isinstance(arithmetic_operator, str):
            raise PipelineSchemaException(f"arithmetic_operator should be a str, found {type(arithmetic_operator)}")
        if not isinstance(column, str):
            raise PipelineSchemaException(f"column should be a str, found {type(column)}")
        if not isinstance(by, float):
            raise PipelineSchemaException(f"by should be a float, found {type(by)}")
        if arithmetic_operator not in ["+","-","*","/"]:
            raise AttributeError('arithmetic_operation must be one of "+","-","*","/"')
        
        self.operations.append({
            "type": "arithmetic",
            "column": column,
            "operation": arithmetic_operator,
            "by": by
        })
        
    def generate_schema(self):
        """Generates Schema for Pipeline Dataset

        Returns:
            dict: Dictionary defining Pipeline Dataset
        """
        if self.create_schema is None:
            raise ValueError("A dataset must have a create operation (create from query/dataset or merge two datasets).")
        
        schema = {
            "name": self.name,
            "create": [self.create_schema]
        }
        schema["create"].extend(self.operations)
        
        return schema
        
class SummaryDataset(Dataset):
    """Pipeline Summary Dataset Definition
    """
    def __init__(self, dataset_name: str, summary_by_row: str, summary_prefix: str = None, summary_suffix: str = None, remove_comma: bool = False, empty_dataset: str = None):
        """Creates Pipeline Summary Dataset Definition

        Args:
            dataset_name (str): Name of Dataset
            summary_by_row (str): Summary statement to generate for each row in dataset
            summary_prefix (str, optional): Prefix for summary statements. Defaults to None.
            summary_suffix (str, optional): Suffix for summary satements. Defaults to None.
            remove_comma (bool, optional): Whether to remove a potential last comma after the last row summary is created. Defaults to False.
            empty_dataset (str, optional): Summary to put in place when dataset is empty. Defaults to None.
        """
        if not isinstance(summary_by_row, str):
            raise PipelineSchemaException(f"summary_by_row should be a str, found {type(summary_by_row)}")
        if summary_prefix is not None and not isinstance(summary_prefix, str):
            raise PipelineSchemaException(f"summary_prefix should be a str, found {type(summary_prefix)}")
        if summary_suffix is not None and not isinstance(summary_suffix, str):
            raise PipelineSchemaException(f"summary_suffix should be a str, found {type(summary_suffix)}")
        if remove_comma is not None and not isinstance(remove_comma, bool):
            raise PipelineSchemaException(f"remove_comma should be a bool, found {type(remove_comma)}")
        if empty_dataset is not None and not isinstance(empty_dataset, str):
            raise PipelineSchemaException(f"empty_dataset should be a str, found {type(empty_dataset)}")
        super().__init__(dataset_name)
        self.prefix = summary_prefix
        self.summary = summary_by_row
        self.suffix = summary_suffix
        self.remove_comma = remove_comma
        self.empty_dataset = empty_dataset
        
    def generate_schema(self):
        """Generates Schema for Pipeline Dataset

        Returns:
            dict: Dictionary defining Pipeline Dataset
        """
        dataset_schema = super().generate_schema()
        if self.prefix is not None:
            dataset_schema["prefix"] = self.prefix
        dataset_schema["summarize"] = self.summary
        if self.suffix is not None:
            dataset_schema["suffix"] = self.suffix
        dataset_schema["remove_comma"] = self.remove_comma
        if self.empty_dataset is not None:
            dataset_schema["empty_summary"] = self.empty_dataset
        return dataset_schema
    
class SemanticCacheDataset(Dataset):
    
    def add_cleanse_cache(self):
        self.operations.append({
            "type": "cleanse_cache"
        })
        
    def add_open_ai_make_cache_comparisons(self, input_field_name: str, open_ai_api_key: str, open_ai_base_url: str = None, open_api_model_name: str = None, similarity_minimum: float = None) -> None:
        """_summary_

        Args:
            input_field_name (str): _description_
            open_ai_api_key (str): _description_
            open_ai_base_url (str, optional): _description_. Defaults to None.
            open_api_model_name (str, optional): _description_. Defaults to None.
            similarity_minimum (float, optional): _description_. Defaults to None.

        Raises:
            PipelineSchemaException: _description_
        """
        if not isinstance(input_field_name, str):
            raise PipelineSchemaException(f"input_field_name should be a str, found {type(input_field_name)}")
        if not isinstance(open_ai_api_key, str):
            raise PipelineSchemaException(f"open_ai_api_key should be a str, found {type(open_ai_api_key)}")
        if open_ai_base_url is not None and not isinstance(open_ai_base_url, str):
            raise PipelineSchemaException(f"open_ai_base_url should be a str, found {type(open_ai_base_url)}")
        if open_api_model_name is not None and not isinstance(open_api_model_name, str):
            raise PipelineSchemaException(f"open_api_model_name should be a str, found {type(open_api_model_name)}")
        if similarity_minimum is not None and not isinstance(similarity_minimum, float):
            raise PipelineSchemaException(f"similarity_minimum should be a float, found {type(similarity_minimum)}")
        operation = {
            "type": "cleanse_cache",
            "params": {
                "field_name": input_field_name,
                "similarity_minimum": open_ai_api_key
            }
        }
        if open_ai_base_url is not None:
            operation["openai_api_base"] = open_ai_base_url
        if open_api_model_name is not None:
            operation["model"] = open_api_model_name
        if similarity_minimum is not None:
            operation["similarity_minimum"] = similarity_minimum
        
        self.operations.append(operation)
    
class AdvancedDataset(SummaryDataset):
    """Pipeline Advanced Dataset Definition
    """
    def __init__(self, dataset_name, summary_by_row = None, summary_prefix = None, summary_suffix = None, remove_comma = False, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Creates Advanced Dataset Definition

        Args:
            dataset_name (str): Name of Dataset
            summary_by_row (str, optional): Summary statement to generate for each row in dataset. Defaults to None.
            summary_prefix (str, optional): Prefix for summary statements. Defaults to None.
            summary_suffix (str, optional): Suffix for summary satements. Defaults to None.
            remove_comma (bool, optional): Whether to remove a potential last comma after the last row summary is created. Defaults to False.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        super().__init__(dataset_name, summary_by_row, summary_prefix, summary_suffix, remove_comma)
        self.required_fields = required_fields
        self.exclude_fields = exclude_fields
    
    def create_from_query(self, query_name: str, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds Create from Query operation to dataset

        Args:
            query_name (str): Name of Query
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.

        Raises:
            ValueError: Adding too many create operations
        """
        if not isinstance(query_name, str):
            raise PipelineSchemaException(f"query_name should be a str, found {type(query_name)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        if self.create_schema is not None:
            operation_dict = {
                "type": "query",
                "name": query_name
            }            
            if required_fields is not None:
                operation_dict["required_fields"] = required_fields
            if exclude_fields is not None:
                operation_dict["exclude_fields"] = exclude_fields
            self.operations.append(operation_dict)
        else:
            self.create_schema = {
                "type": "query",
                "name": query_name
            }
            if required_fields is not None:
                self.create_schema["required_fields"] = required_fields
            if exclude_fields is not None:
                self.create_schema["exclude_fields"] = exclude_fields
    
    def create_from_dataset(self, dataset_name: str, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds Create from Query operation to dataset

        Args:
            dataset_name (str): Name o Dataset
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        if not isinstance(dataset_name, str):
            raise PipelineSchemaException(f"dataset_name should be a str, found {type(dataset_name)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        if self.create_schema is not None:
            operation_dict = {
                "type": "datasaet",
                "name": dataset_name
            }
            if required_fields is not None:
                operation_dict["required_fields"] = required_fields
            if exclude_fields is not None:
                operation_dict["exclude_fields"] = exclude_fields
            self.operations.append(operation_dict)
        else:
            self.create_schema = {
                "type": "dataset",
                "name": dataset_name
            }
            if required_fields is not None:
                self.create_schema["required_fields"] = required_fields
            if exclude_fields is not None:
                self.create_schema["exclude_fields"] = exclude_fields
        
    def merge_two_datasets(self, dataset1_name: str, dataset2_name: str, how: str, left_on: str, right_on: str, nan_replace: Any = None, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds Create from Merge operation to dataset

        Args:
            dataset1_name (str): Name of Left Dataset
            dataset2_name (str): Name of Right Dataset
            how (str): How to merge [left,right,inner,outer].
            left_on (str): Column in dataset 1 to use for merge
            right_on (str): Column in dataset 2 to use for merge
            nan_replace (Any, optional): Value to replace nan values with. If None does not replace nan values. Defaults to None.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.

        Raises:
            AttributeError: How must be one of [left,right,inner,outer]
        """
        if not isinstance(dataset1_name, str):
            raise PipelineSchemaException(f"dataset1_name should be a str, found {type(dataset1_name)}")
        if not isinstance(dataset2_name, str):
            raise PipelineSchemaException(f"dataset2_name should be a str, found {type(dataset2_name)}")
        if not isinstance(how, str):
            raise PipelineSchemaException(f"how should be a str, found {type(how)}")
        if not isinstance(left_on, str):
            raise PipelineSchemaException(f"left_on should be a str, found {type(left_on)}")
        if not isinstance(right_on, str):
            raise PipelineSchemaException(f"right_on should be a str, found {type(right_on)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        how = how.lower()
        if how not in ["left","right","inner","outer"]:
            raise AttributeError('how must be one of "left","right","inner","outer"')
        
        if self.create_schema is not None:
            operation_dict = {
                "type": "merge",
                "dataset1": dataset1_name,
                "dataset2": dataset2_name,
                "how": how,
                "left_on": left_on,
                "right_on": right_on
            }
            if nan_replace is not None:
                operation_dict["nan_replace"] = nan_replace
            if required_fields is not None:
                operation_dict["required_fields"] = required_fields
            if exclude_fields is not None:
                operation_dict["exclude_fields"] = exclude_fields
            self.operations.append(operation_dict)
        else:        
            self.create_schema = {
                "type": "merge",
                "dataset1": dataset1_name,
                "dataset2": dataset2_name,
                "how": how,
                "left_on": left_on,
                "right_on": right_on
            }
            if nan_replace is not None:
                self.create_schema["nan_replace"] = nan_replace
            if required_fields is not None:
                self.create_schema["required_fields"] = required_fields
            if exclude_fields is not None:
                self.create_schema["exclude_fields"] = exclude_fields
        
    def add_function(self, function_name: str, function_fields_dict: dict = None, function_params_dict: dict = None, function_environment_variables_dict: dict = None, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds function to Dataset Operations

        Args:
            function_name (str): Name of function in Pipelines functions dict
            function_fields_dict (dict, optional): Dictionary of Fields to map to parameters for the function. If None is excluded from schema. Defaults to None.
            function_params_dict (dict, optional): Dictionary of static parameters to map to parameters for the function. If None is excluded from schema. Defaults to None.
            function_environment_variables_dict (dict, optional): Dictionary of environment variables to map to parameters for the function. If None is excluded from schema. Defaults to None.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        if not isinstance(function_name, str):
            raise PipelineSchemaException(f"function_name should be a str, found {type(function_name)}")
        if function_fields_dict is not None and not isinstance(function_fields_dict, list):
            raise PipelineSchemaException(f"function_fields_dict should be a list, found {type(function_fields_dict)}")
        if function_params_dict is not None and not isinstance(function_params_dict, list):
            raise PipelineSchemaException(f"function_params_dict should be a list, found {type(function_params_dict)}")
        if function_environment_variables_dict is not None and not isinstance(function_environment_variables_dict, list):
            raise PipelineSchemaException(f"function_environment_variables_dict should be a list, found {type(function_environment_variables_dict)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        function_dict = {
            "type" : "function",
            "name" : function_name
        }
            
        if function_fields_dict is not None:
            function_dict["fields"] = function_fields_dict
            
        if function_params_dict is not None:
            function_dict["params"] = function_params_dict
        
        if function_params_dict is not None:
            function_dict["environment"] = function_environment_variables_dict
            
        if required_fields is not None:
            function_dict["required_fields"] = required_fields
            
        if exclude_fields is not None:
            function_dict["exclude_fields"] = exclude_fields
            
        self.operations.append(function_dict)
    
    def add_filter(self, filters_list: list[str], required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds filter to Dataset Operations

        Args:
            filters_list (list[str]): List of Filters to apply
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        if not isinstance(filters_list, list):
            raise PipelineSchemaException(f"filters_list should be a list, found {type(filters_list)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        operation_dict = {
            "type": "filter",
            "filters": filters_list
        }
        if required_fields is not None:
            operation_dict["required_fields"] = required_fields
        if exclude_fields is not None:
            operation_dict["exclude_fields"] = exclude_fields
        self.operations.append(operation_dict)
        
    def add_arithmetic_operation(self, arithmetic_operator: str, column: str, by: float, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds Arithmetic Operation to Dataset Operations

        Args:
            arithmetic_operator (str): Arithmetic Operator
            column (str): Name of Column to apply operator to
            by (float): value to apply operation by
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.

        Raises:
            AttributeError: _description_
        """
        if not isinstance(arithmetic_operator, str):
            raise PipelineSchemaException(f"arithmetic_operator should be a str, found {type(arithmetic_operator)}")
        if not isinstance(column, str):
            raise PipelineSchemaException(f"column should be a str, found {type(column)}")
        if not isinstance(by, float):
            raise PipelineSchemaException(f"by should be a float, found {type(by)}")
        if required_fields is not None and not isinstance(required_fields, list):
            raise PipelineSchemaException(f"required_fields should be a list, found {type(required_fields)}")
        if exclude_fields is not None and not isinstance(exclude_fields, list):
            raise PipelineSchemaException(f"exclude_fields should be a list, found {type(exclude_fields)}")
        if arithmetic_operator not in ["+","-","*","/"]:
            raise AttributeError('arithmetic_operation must be one of "+","-","*","/"')
        
        operation_dict = {
            "type": "arithmetic",
            "column": column,
            "operation": arithmetic_operator,
            "by": by
        }
        if required_fields is not None:
            operation_dict["required_fields"] = required_fields
        if exclude_fields is not None:
            operation_dict["exclude_fields"] = exclude_fields        
        self.operations.append(operation_dict)
        
    def generate_schema(self):
        """Generates Schema for Pipeline Advanced Dataset

        Returns:
            dict: Dictionary defining Pipeline Advanced Dataset
        """
        dataset_schema = super().generate_schema()
        if self.summary is None:
            del dataset_schema["summary"]
            del dataset_schema["remove_comma"]
        if self.required_fields is not None:
            dataset_schema["required_fields"] = self.required_fields
        if self.exclude_fields is not None:
            dataset_schema["exclude_fields"] = self.exclude_fields
        
class Summary(PipelineComponent):
    """Pipeline Summary Definition
    """
    def __init__(self, datasets: list[SummaryDataset], prefix = None, suffix = None):
        """Creates Pipeline Summary Definition

        Args:
            datasets (list[SummaryDataset]): List of datasets to build summary from
            prefix (str): Text for the prefix of the summary
            suffix (str): Text for the suffix of the summary
        """
        if not isinstance(datasets, list):
            raise PipelineSchemaException(f"datasets should be a list, found {type(datasets)}")
        if not isinstance(prefix, str):
            raise PipelineSchemaException(f"prefix should be a str, found {type(prefix)}")
        if not isinstance(suffix, str):
            raise PipelineSchemaException(f"suffix should be a str, found {type(suffix)}")
        self.datasets = datasets
        self.prefix = prefix
        self.suffix = suffix
        
    def generate_schema(self):
        """Generates Schema for Pipeline Summary

        Returns:
            dict: Dictionary defining Pipeline Summary
        """
        if len(self.datasets) == 0:
            raise ValueError("There must be at least one dataset to generate a summary.")
        
        summary_schema = {
            "datasets": []
        }        
        for dataset in self.datasets:
            summary_schema["datasets"].append(dataset.get_name())
            
        if self.prefix is not None:
            summary_schema["prefix"] = self.prefix
            
        if self.suffix is not None:
            summary_schema["suffix"] = self.suffix
        
        return summary_schema
        
class Visualization(PipelineComponent):
    """Pipeline Visualization Definition
    """    
    def __init__(self, dataset: Dataset, title: str, description: str, tooltip: bool = True):
        """Creates Pipeline Visualization Definition

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Visualization Title
            description (str): Description of Visualization
        """
        if not isinstance(dataset, Dataset):
            raise PipelineSchemaException(f"dataset should be a Dataset, found {type(dataset)}")
        if not isinstance(title, str):
            raise PipelineSchemaException(f"title should be a str, found {type(title)}")
        if not isinstance(description, str):
            raise PipelineSchemaException(f"description should be a str, found {type(description)}")
        if not isinstance(tooltip, bool):
            raise PipelineSchemaException(f"tooltip should be a bool, found {type(tooltip)}")
        self.dataset = dataset.get_name()
        self.title = title
        self.description = description
        self.tooltip = tooltip
        
class PieChart(Visualization):
    """Pipeline Pie Chart Definition
    """    
    def __init__(self, dataset: Dataset, title: str, description: str, value_column: str, label_column: str, tooltip: bool = True):
        """Creates Pipeline Pie Chart Defintion

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Pie Chart Title
            description (str): Description of Pie Chart
            value_column (str): Name of column to that has the size of each pie slice
            label_column (str): Name of column to that has the name of each pie slice
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(value_column, str):
            raise PipelineSchemaException(f"value_column should be a str, found {type(value_column)}")
        if not isinstance(label_column, str):
            raise PipelineSchemaException(f"label_column should be a str, found {type(label_column)}")
        super().__init__(dataset, title, description, tooltip)
        self.value_column = value_column
        self.label_column = label_column
        
    def generate_schema(self):
        """Generates Schema for Pipeline Pie Chart

        Returns:
            dict: Dictionary defining Pipeline Pie Chart
        """
        return {
            "type": "pie",
            "dataset": self.dataset,
            "title": self.title,
            "value_column": self.value_column,
            "label_column": self.label_column,
            "description": self.description,
            "tooltip": self.tooltip
        }
        
class LineGraph(Visualization):
    """Pipeline Line Graph Definition
    """    
    def __init__(self, dataset: Dataset, title: str, description: str, x_axis: str, y_axis: str, tooltip: bool = True):
        """Creates Pipeline Line Graph Defintion

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            x_axis (str): Column Name to use for x_axis
            y_axis (str): Column Name to use for y_axis
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(x_axis, str):
            raise PipelineSchemaException(f"x_axis should be a str, found {type(x_axis)}")
        if not isinstance(y_axis, str):
            raise PipelineSchemaException(f"y_axis should be a str, found {type(y_axis)}")
        super().__init__(dataset, title, description, tooltip)
        self.x_axis = x_axis
        self.y_axis = y_axis
        
    def generate_schema(self):
        """Generates Schema for Pipeline Line Graph

        Returns:
            dict: Dictionary defining Pipeline Line Graph
        """
        return {
            "type": "line",
            "dataset": self.dataset,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "description": self.description,
            "tooltip": self.tooltip
        }

class MultiLineGraph(Visualization):
    """Pipeline Multi Line Graph Definition. Use if one column are y-values and one column indicates which line the y-value is a part of.
    """    
    def __init__(self, dataset: Dataset, title: str, description: str, x_axis: str, y_axis: str, color_column: str, tooltip: bool = True):
        """Creates Pipeline Multi Line Graph Defintion

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            x_axis (str): Column Name to use for x_axis
            y_axis (str): Column Name to use for y_axis
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(x_axis, str):
            raise PipelineSchemaException(f"x_axis should be a str, found {type(x_axis)}")
        if not isinstance(y_axis, str):
            raise PipelineSchemaException(f"y_axis should be a str, found {type(y_axis)}")
        super().__init__(dataset, title, description, tooltip)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.color_column = color_column
        
    def generate_schema(self):
        """Generates Schema for Pipeline Multi Line Graph

        Returns:
            dict: Dictionary defining Pipeline Multi Line Graph
        """
        return {
            "type": "line",
            "dataset": self.dataset,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "color_column": self.color_column,
            "description": self.description,
            "tooltip": self.tooltip
        }
        
class MultiColumnLineGraph(Visualization):
    """Pipeline Multi-Column Line Graph Definition. Use if each column should be a line.
    """    
    def __init__(self, dataset: Dataset, title: str, description: str, x_axis: str, columns: list[str], y_axis: str, tooltip: bool = True):
        """Creates Pipeline Multi-Line Graph Defintion

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            x_axis (str): Column Name to use for x_axis
            columns (list[str]): List of column names to use for y_axis
            y_axis (str): Axis title for y_axis
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(x_axis, str):
            raise PipelineSchemaException(f"x_axis should be a str, found {type(x_axis)}")
        if not isinstance(columns, list):
            raise PipelineSchemaException(f"columns should be a list, found {type(columns)}")
        if not isinstance(y_axis, str):
            raise PipelineSchemaException(f"y_axis should be a str, found {type(y_axis)}")
        super().__init__(dataset, title, description, tooltip)
        self.x_axis = x_axis
        self.y_axis = columns
        self.y_axis_name = y_axis
        
    def generate_schema(self):
        """Generates Schema for Pipeline Multi-Line Graph

        Returns:
            dict: Dictionary defining Pipeline Multi-Line Graph
        """
        return {
            "type": "line",
            "dataset": self.dataset,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "y_axis_name": self.y_axis_name,
            "description": self.description,
            "tooltip": self.tooltip
        }
        
class Histogram(Visualization):
    """Pipeline Histogram Definition
    """
    def __init__(self, dataset: Dataset, title: str, description: str, value_column: str, tooltip: bool = True):
        """Creates Pipeline Histogram Definition

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            value_column (str): Column from dataset with values for histogram
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(value_column, str):
            raise PipelineSchemaException(f"value_column should be a str, found {type(value_column)}")
        super().__init__(dataset, title, description, tooltip)
        self.value_column = value_column
        
    def generate_schema(self):
        """Generates Schema for Pipeline Histogram

        Returns:
            dict: Dictionary defining Pipeline Histogram
        """
        return {
            "type": "histogram",
            "dataset": self.dataset,
            "title": self.title,
            "value_column": self.value_column,
            "description": self.description,
            "tooltip": self.tooltip
        }
        
class BarChart(Visualization):
    """Pipeline Bar Chart Definition
    """
    def __init__(self, dataset: Dataset, title: str, description: str, value_column: str, label_column: str, tooltip: bool = True):
        """Creates Pipeline Bar Chart Definition

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            value_column (str): Name of column with stacked bar size
            label_column (str): Name of column with indexes for x_axis
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(value_column, str):
            raise PipelineSchemaException(f"value_column should be a str, found {type(value_column)}")
        if not isinstance(label_column, str):
            raise PipelineSchemaException(f"label_column should be a str, found {type(label_column)}")
        super().__init__(dataset, title, description, tooltip)
        self.value_column = value_column
        self.label_column = label_column
        
    def generate_schema(self):
        return {
            "type": "bar",
            "dataset": self.dataset,
            "title": self.title,
            "value_column": self.value_column,
            "label_column": self.label_column,
            "description": self.description,
            "tooltip": self.tooltip
        }
        
class StackedBarChart(Visualization):
    """Pipeline Stacked Bar Chart Definition
    """
    def __init__(self, dataset: Dataset, title: str, description: str, value_column: str, index_column: str, color_column: str, tooltip: bool = True):
        """Creates Pipeline Stacked Bar Chart Definition

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            value_column (str): Name of column with stacked bar size
            index_column (str): Name of column with indexes for x_axis
            color_column (str): Name of column with label for each stacked bar
            tooltip (bool): Whether tooltips should be shown
        """
        if not isinstance(value_column, str):
            raise PipelineSchemaException(f"value_column should be a str, found {type(value_column)}")
        if not isinstance(index_column, str):
            raise PipelineSchemaException(f"index_column should be a str, found {type(index_column)}")
        if not isinstance(color_column, str):
            raise PipelineSchemaException(f"color_column should be a str, found {type(color_column)}")
        super().__init__(dataset, title, description, tooltip)
        self.value_column = value_column
        self.index_column = index_column
        self.color_column = color_column
        
    def generate_schema(self):
        return {
            "type": "stacked_bar",
            "dataset": self.dataset,
            "title": self.title,
            "value_column": self.value_column,
            "index_column": self.index_column,
            "color_column": self.color_column,
            "description": self.description,
            "tooltip": self.tooltip
        }
    
class Action(PipelineComponent):
    """Action class for executing a python function
    """
    def __init__(self, name: str, function_name: str, output_field: Union[str,None] = None):
        """Creates an Action Schema

        Args:
            name (str): Name of the action
            function_name (str): Name of the function in the function dict passed to the Pipeline
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        if not isinstance(function_name, str):
            raise PipelineSchemaException(f"function_name should be a str, found {type(function_name)}")
        if output_field is not None and not isinstance(output_field, str):
            raise PipelineSchemaException(f"output_field should be a str, found {type(output_field)}")
        self.command_name = name
        self.output_field = output_field
        self.attributes = []
        self.function_name = function_name
        self.params = {}
        self.connectors = []
        self.field_strings = {}
        
    def add_attribute(self, name: str, field: Union[str,PipelineField], optional: bool = True) -> None:
        """Adds an attribute that will allow your command to include a field's value as an argument to the python function

        Args:
            name (str): Name of the attribute
            field (str): The name of the field to get the value for the attribute from
            optional (bool): Whether the attribute is optional for the action to run
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        if not isinstance(field, str) and not isinstance(field, PipelineField):
            raise PipelineSchemaException(f"field should be a str or PipelineField, found {type(name)}")
        if optional is not None and not isinstance(optional, bool):
            raise PipelineSchemaException(f"optional should be a bool, found {type(optional)}")
        name = name if isinstance(name,str) else name.get_section() + "." + name.get_name()
        attribute_dict = {
            "name": name,
            "field": field,
            "optional": optional
        }
        self.attributes.append(attribute_dict)
        
    def add_param(self, name: str, value: Any) -> None:
        """Adds a parameter to the Python Function. This is a static value that will be passed to your python function as an argument.

        Args:
            name (str): Parameter name
            value (Any): Parameter value
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        self.params[name] = value
        
    def add_connector(self, connector_name: str, param_name: str) -> None:
        """Adds a connector to be passed to the function associated with this action

        Args:
            connector_name (str): Name of the connector
            param_name (str): Name of the parameter in the function associated with this action
        """
        if not isinstance(connector_name, str):
            raise PipelineSchemaException(f"connector_name should be a str, found {type(connector_name)}")
        if not isinstance(param_name, str):
            raise PipelineSchemaException(f"param_name should be a str, found {type(param_name)}")
        self.connectors.append({
            "name": connector_name,
            "param": param_name
        })
        
    def add_field_string(self, name: str, field_string: str):
        """Adds a field string to the command. Ex. if field_string = 'The number is {base.number}' and there is a field, 
        base.number = 5, then the value passed to the action's function would be 'The number is 5'.

        Args:
            name (str): Name of the parameter in the function
            field_string (str): The string to populate with fields
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        if not isinstance(field_string, str):
            raise PipelineSchemaException(f"field_string should be a str, found {type(field_string)}")
        self.field_strings[name] = field_string

    def get_name(self) -> str:
        """Gets the action name

        Returns:
            str: Action Name
        """
        return self.command_name
        
    def generate_schema(self) -> dict:
        schema = {
            "name": self.command_name
        }
        schema["function"] = self.function_name
        if len(self.attributes.keys()) > 0:
            schema["attributes"] = self.attributes
        if len(self.params) > 0:
            schema["params"] = self.params
        if len(self.connectors) > 0:
            schema["connectors"] = self.connectors
        if len(self.field_strings) > 0:
            schema["field_strings"] = self.field_strings
        if self.output_field is not None:
            schema["output_field"] = self.output_field
        return schema
    
class ConsoleAction(Action):
    """Action class for building a console/terminal command
    """
    
    def __init__(self, name: str, output_field: Union[str,None] = None):
        """Creates a ConsoleAction Schema. This action is used specifically generate a str output that can be executed in a console/termianl.

        Args:
            name (str): name of the action
            output_field (Union[str,None], optional): Output field to store result in. Defaults to None.
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        if output_field is not None and not isinstance(output_field, str):
            raise PipelineSchemaException(f"output_field should be a str, found {type(output_field)}")
        super().__init__(name, "generate_console_command", output_field)
        
    def add_attribute(self, name: str, field: str, optional: bool = True, tag: Union[str,None] = None) -> None:
        """Adds an attribute to the Console Action

        Args:
            name (str): Name of the attribute
            field (str): The name of the field to get the value for the attribute from
            optional (bool): Whether the attribute is optional for the action to run
            tag (Union[str,None], optional): Value to use for the tag in the console, I.E. --help or -h. If None, dsl-spa uses --{attribute_name}. Defaults to None.
        """
        if tag is not None and not isinstance(tag, str):
            raise PipelineSchemaException(f"tag should be a str, found {type(tag)}")
        super().add_attribute(name, field, optional)
        if tag is not None:
            self.attributes[name]["tag"] = tag
            
class Command(PipelineComponent):
    """Command class for defining sequence of actions
    """
    
    def __init__(self, name: str, actions: list[Action]):
        """Creates a Command Schema

        Args:
            name (str): name of the command
            actions (list[Action]): list of actions that define the command
        """
        if not isinstance(name, str):
            raise PipelineSchemaException(f"name should be a str, found {type(name)}")
        if not isinstance(actions, list):
            raise PipelineSchemaException(f"actions should be a list, found {type(actions)}")
        self.command_name = name
        self.actions = actions
        
    def get_name(self) -> str:
        """Gets the name of the command

        Returns:
            str: Name of the command
        """
        return self.command_name
        
    def generate_schema(self):
        schema = list(map(lambda x: x.get_name(), self.actions))
        return schema
        
class PipelineSchema:
    """Creates Schema for Pipeline
    """    
    def __init__(self, pipeline_name: str, fields: list[PipelineField]):
        """Creates Schema for Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
        """
        if not isinstance(pipeline_name, str):
            raise PipelineSchemaException(f"pipeline_name should be a str, found {type(pipeline_name)}")
        if not isinstance(fields, list):
            raise PipelineSchemaException(f"fields should be a list, found {type(fields)}")
        self.name = pipeline_name
        self.fields = fields
        self.schema = {
            "pipeline_name": self.name
        }
        
    def build_pipeline_schema(self):
        """Builds Pipeline Schema
        """
        self.build_fields_schema()
    
    def build_fields_schema(self):
        """Builds Fields Schema
        """
        fields_schema = {}
        for field in self.fields:
            section = field.get_section()
            sections = section.split('.')
            sub_schema = fields_schema
            for sect in sections:
                if section not in sub_schema.keys():
                    sub_schema[sect] = {}
                sub_schema = sub_schema[sect]
            sub_schema[field.get_name()] = field.generate_schema()
            
        self.schema["fields"] = fields_schema
            
    def get_schema(self) -> dict:
        """Gets Pipeline Schema

        Returns:
            dict: Schema of Pipeline
        """
        self.build_pipeline_schema()
        return self.schema
    
class CommandPipelineSchema(PipelineSchema):
    
    def __init__(self, pipeline_name: str, fields: list[PipelineField], actions: list[Action], commands: list[Command]):
        if not isinstance(actions, list):
            raise PipelineSchemaException(f"actions should be a list, found {type(actions)}")
        if not isinstance(commands, list):
            raise PipelineSchemaException(f"commands should be a list, found {type(commands)}")
        super().__init__(pipeline_name, fields)
        self.actions = actions
        self.commands = commands
        
    def build_action_schema(self) -> None:
        """Builds Action Schema
        """
        self.schema["actions"] = []
        for command in self.actions:
            schema = command.generate_schema()
            self.schema["actions"].append(schema)
        
    def build_command_schema(self) -> None:
        """Builds Command Schema
        """
        self.schema["commands"] = {}
        for command_sequence in self.commands:
            self.schema["commands"][command_sequence.get_name()] = command_sequence.generate_schema()
            
    def build_pipeline_schema(self) -> None:
        """Builds Command Pipeline Schema
        """
        super().build_pipeline_schema()
        self.build_action_schema()
        self.build_command_schema()
        
class BasicPipelineSchema(PipelineSchema):
    """Creates Schema for Basic Pipeline
    """    
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[Query], csvs: list[CSV], datasets: list[Dataset]):
        """Creates Schema for Basic Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[Query]): List of Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
        """
        if not isinstance(queries, list):
            raise PipelineSchemaException(f"queries should be a list, found {type(queries)}")
        if not isinstance(csvs, list):
            raise PipelineSchemaException(f"csvs should be a list, found {type(csvs)}")
        if not isinstance(datasets, list):
            raise PipelineSchemaException(f"datasets should be a list, found {type(datasets)}")
        super().__init__(pipeline_name, fields)
        self.queries = queries
        self.csvs = csvs
        self.filters: list[Filter] = None
        self.datasets = datasets
        self.schema = {
            "pipeline_name": self.name
        }
        
    def build_pipeline_schema(self):
        """Builds Basic Pipeline Schema
        """
        super().build_pipeline_schema()
        self.build_queries_schema()
        self.build_csvs_schema()
        self.build_dataset_schema()
        
    def build_queries_schema(self):
        """Builds Queries Schema
        """
        if len(self.queries) > 0:
            self.schema["queries"] = {}
        for query in self.queries:
            self.schema["queries"][query.get_name()] = query.generate_schema()
    
    def build_csvs_schema(self):
        if len(self.csvs) > 0:
            self.schema["csvs"] = []
        for csv in self.csvs:
            self.schema["csvs"].append(csv.generate_schema())
            
    def build_filters_schema(self):
        """Builds Filters Schema
        """
        if self.filters is None:
            return
        if len(self.filters) > 0:
            self.schema["filters"] = []
        for pipeline_filter in self.filters:
            self.schema["filters"].append(pipeline_filter.generate_schema())
            
    def build_dataset_schema(self):
        """Builds Datasets Schema
        """
        if len(self.datasets) > 0:
            self.schema["datasets"] = []
        for dataset in self.datasets:
            self.schema["datasets"].append(dataset.generate_schema())
            
class StandardPipelineSchema(BasicPipelineSchema):
    """Creates Schema for Standard Pipeline
    """    
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[Query], csvs: list[CSV], datasets: list[Dataset], scope: str, scope_description: str, summary: Summary = None, visualizations: list[Visualization] = None):
        """Creates Schema for Standard Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[Query]): List of Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
            scope (str): Definition of scope of the pipeline
            scope_description (str): Description of scope of the pipeline
            summary (Summary, optional): Summary for pipeline. Defaults to None.
            visualizations (list[Visualization], optional): Visualizations for pipeline. Defaults to None.
        """
        if not isinstance(scope, str):
            raise PipelineSchemaException(f"scope should be a str, found {type(scope)}")
        if not isinstance(scope_description, str):
            raise PipelineSchemaException(f"scope_description should be a str, found {type(scope_description)}")
        if summary is not None and not isinstance(summary, Summary):
            raise PipelineSchemaException(f"summary should be a Summary, found {type(summary)}")
        if visualizations is not None and not isinstance(visualizations, list):
            raise PipelineSchemaException(f"visualizations should be a list, found {type(visualizations)}")
        super().__init__(pipeline_name, fields, queries, csvs, datasets)
        self.scope = scope
        self.scope_description = scope_description
        self.summary = summary
        self.visualizations = visualizations
        
    def build_pipeline_schema(self):
        """Builds Standard Pipeline Schema
        """
        self.schema["scope"] = self.scope
        self.schema["scope_description"] = self.scope_description
        super().build_pipeline_schema()
        if self.summary is not None:
            self.build_summary_schema()
            
        if self.visualizations is not None and len(self.visualizations) > 0:
            self.build_visualizations_schema()
        
    def build_summary_schema(self):
        """Builds Summary Schema
        """
        self.schema["summary"] = self.summary.generate_schema()
        
    def build_visualizations_schema(self):
        """Builds Visualizations Schema
        """
        self.schema["visualizations"] = []
        for visualization in self.visualizations:
            self.schema["visualizations"].append(visualization.generate_schema())
            
class DashboardPipelineSchema(StandardPipelineSchema):
    """Creates Schema for Dashboard Pipeline
    """
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[Query], csvs: list[CSV], filters: list[Filter], datasets: list[Dataset], scope: str, scope_description: str, summary: Summary = None, visualizations: list[Visualization] = None):
        """Creates Schema for Dashboard Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[Query]): List of Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            filters (list[Filter]): List of Filters for pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
            scope (str): Definition of scope of the pipeline
            scope_description (str): Description of scope of the pipeline
            summary (Summary, optional): Summary for pipeline. Defaults to None.
            visualizations (list[Visualization], optional): Visualizations for pipeline. Defaults to None.
        """
        if not isinstance(filters, list):
            raise PipelineSchemaException(f"filters should be a list, found {type(filters)}")
        super().__init__(pipeline_name, fields, queries, csvs, datasets, scope, scope_description, summary, visualizations)
        self.filters = filters
        
    def build_pipeline_schema(self):
        """Builds Dashboard Pipeline Schema
        """
        super().build_pipeline_schema()
        super().build_filters_schema()
        
class SemanticCachePipelineSchema(BasicPipelineSchema):
    """Creates Schema for Semantic Cache Pipeline
    """
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[Query], csvs: list[CSV], datasets: list[Dataset], semantic_cache_dataset: str, results_columns: list[str], empty_cache_error_message: str = None):
        """Creates Schema for Semantic Cache Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[Query]): List of Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
            semantic_cache_dataset (str): Name of dataset housing semantic cache
            results_columns (list[str]): Name of Columns to be output from semantic cache when getting results
            empty_cache_error_message (str, optional): Error message to throw when the semantic cache is empty after processing. Defaults to None.
        """
        if not isinstance(semantic_cache_dataset, str):
            raise PipelineSchemaException(f"semantic_cache_dataset should be a str, found {type(semantic_cache_dataset)}")
        if not isinstance(results_columns, list):
            raise PipelineSchemaException(f"results_columns should be a list, found {type(results_columns)}")
        if empty_cache_error_message is not None and not isinstance(empty_cache_error_message, str):
            raise PipelineSchemaException(f"empty_cache_error_message should be a str, found {type(empty_cache_error_message)}")
        super().__init__(pipeline_name, fields, queries, csvs, datasets)
        self.semantic_cache_dataset = semantic_cache_dataset
        self.results_columns = results_columns
        self.empty_cache_error_message = empty_cache_error_message
        
    def build_pipeline_schema(self):
        """Builds Semantic Cache Pipeline Schema
        """
        super().build_pipeline_schema()
        self.build_semantic_cache_schema()
        
    def build_semantic_cache_schema(self):
        """Builds Semantic Cache Schema
        """
        super().build_pipeline_schema()
        semantic_cache_schema = {
            "dataset": self.semantic_cache_dataset,
            "results_columns": self.results_columns
        }
        if self.empty_cache_error_message is not None:
            semantic_cache_schema["empty_cache_error_message"] = self.empty_cache_error_message
        self.schema["semantic_cache"] = semantic_cache_schema