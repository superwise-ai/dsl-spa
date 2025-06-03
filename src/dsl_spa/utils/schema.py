from typing import Any

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

class SQLQuery(PipelineComponent):
    """SQL Query Definition
    """
    def __init__(self, query_name: str, connector_name: str):
        """Creates SQL Query Definition

        Args:
            query_name (str): Name of Query
            connector_name (str): Name of Connector
        """
        self.name = query_name
        self.connector = connector_name
        self.clauses = []
    
    def get_name(self) -> str:
        """Gets Query Name

        Returns:
            str: Query Name
        """
        return self.name
        
    def add_clause(self, sql_clause: str, optional: bool, field_required: str = None):
        """Adds SQL Clause to SQL definition

        Args:
            sql_clause (str): sql clause to add
            optional (bool): whether clause is optional
            field_required (str, optional): The required field to run. Set this to none unless optional is true. Defaults to None.
        """
        clause = {
            "sql": sql_clause,
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
            "sql_clauses": self.clauses
        }
        return query_dict
        
class AdvancedSQLQuery(SQLQuery):
    """Advanced SQL Query Definition
    """
    def __init__(self, query_name: str, connector_name: str, min_results: int = None, error_message: str = None, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Creates an Advanced SQL Query Definition

        Args:
            query_name (str): Name of Query
            connector_name (str): Name of Connector
            min_results (int, optional): Minimum results query must have to not cause an error. If None is excluded from schema. Defaults to None.
            error_message (str, optional): Error message to display if minimum results requirement is not met. If None is excluded. Defaults to None.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        super().__init__(query_name,connector_name)
        self.min_results = min_results
        self.error_message = error_message
        self.required_fields = required_fields
        self.exclude_fields = exclude_fields
        
    def generate_schema(self) -> dict:        
        """Generates Schema for Pipeline Advanced SQL Query

        Returns:
            dict: Dictionary defining Pipeline Advanced SQL Query
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
        if self.create_schema is not None:
            raise ValueError("A dataset can only have one create operation (create from query/dataset or merge two datasets).")
        
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
    def __init__(self, dataset_name: str, summary_by_row: str, summary_prefix: str = None, summary_suffix: str = None, remove_comma: bool = False):
        """Creates Pipeline Summary Dataset Definition

        Args:
            dataset_name (str): Name of Dataset
            summary_by_row (str): Summary statement to generate for each row in dataset
            summary_prefix (str, optional): Prefix for summary statements. Defaults to None.
            summary_suffix (str, optional): Suffix for summary satements. Defaults to None.
            remove_comma (bool, optional): Whether to remove a potential last comma after the last row summary is created. Defaults to False.
        """
        super().__init__(dataset_name)
        self.prefix = summary_prefix
        self.summary = summary_by_row
        self.suffix = summary_suffix
        self.remove_comma = remove_comma
        
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
        return dataset_schema
    
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
        
    def add_function(self, function_name: str, function_fields_dict: dict = None, function_params_dict: dict = None, required_fields: list[list[str]] = None, exclude_fields: list[list[str]] = None):
        """Adds function to Dataset Operations

        Args:
            function_name (str): Name of function in Pipelines functions dict
            function_fields_dict (dict, optional): Dictionary of Fields to map to parameters for the function. If None is excluded from schema. Defaults to None.
            function_params_dict (dict, optional): Dictionary of static parameters to map to parameters for the function. If None is excluded from schema. Defaults to None.
            required_fields (list[list[str]], optional): Fields Required to run query. If None is excluded. Defaults to None.
            exclude_fields (list[list[str]], optional): Fields that would indicate to exclude running the query. If None is excluded. Defaults to None.
        """
        function_dict = {
            "type" : "function",
            "name" : function_name
        }
            
        if function_fields_dict is not None:
            function_dict["fields"] = function_fields_dict
            
        if function_params_dict is not None:
            function_dict["params"] = function_params_dict
            
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
    def __init__(self, datasets: list[SummaryDataset]):
        """Creates Pipeline Summary Definition

        Args:
            datasets (list[SummaryDataset]): List of datasets to build summary from
        """
        self.datasets = datasets
        
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
    """Pipeline Multi-Line Graph Definition
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
    def __init__(self, dataset: Dataset, title: str, description: str, value_column: str, index_column: str, color_column: str, tooltip: bool = True):
        """Creates Pipeline Bar Chart Definition

        Args:
            dataset (Dataset): Dataset to visualize
            title (str): Line Graph Title
            description (str): Description of Line Graph
            value_column (str): Name of column with stacked bar size
            index_column (str): Name of column with indexes for x_axis
            tooltip (bool): Whether tooltips should be shown
        """
        super().__init__(dataset, title, description, tooltip)
        self.value_column = value_column
        self.index_column = index_column
        
    def generate_schema(self):
        return {
            "type": "bar",
            "dataset": self.dataset,
            "title": self.title,
            "value_column": self.value_column,
            "index_column": self.index_column,
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
        
class PipelineSchema:
    """Creates Schema for Pipeline
    """    
    def __init__(self, pipeline_name: str, fields: list[PipelineField]):
        """Creates Schema for Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
        """
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
        
class BasicPipelineSchema(PipelineSchema):
    """Creates Schema for Basic Pipeline
    """    
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[SQLQuery], csvs: list[CSV], datasets: list[Dataset]):
        """Creates Schema for Basic Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[SQLQuery]): List of SQL Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
        """
        self.name = pipeline_name
        self.fields = fields
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
        self.build_fields_schema()
        self.build_queries_schema()
        self.build_csvs_schema()
        self.build_dataset_schema()
    
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
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[SQLQuery], csvs: list[CSV], datasets: list[Dataset], scope: str, scope_description: str, summary: Summary = None, visualizations: list[Visualization] = None):
        """Creates Schema for Standard Pipeline

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[SQLQuery]): List of SQL Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
            scope (str): Definition of scope of the pipeline
            scope_description (str): Description of scope of the pipeline
            summary (Summary, optional): Summary for pipeline. Defaults to None.
            visualizations (list[Visualization], optional): Visualizations for pipeline. Defaults to None.
        """
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
    def __init__(self, pipeline_name: str, fields: list[PipelineField], queries: list[SQLQuery], csvs: list[CSV], filters: list[Filter], datasets: list[Dataset], scope: str, scope_description: str, summary: Summary = None, visualizations: list[Visualization] = None):
        """_summary_

        Args:
            pipeline_name (str): Pipeline Name
            fields (list[PipelineField]): List of fields for pipeline
            queries (list[SQLQuery]): List of SQL Queries for pipeline
            csvs (list[CSV]): List of CSVs for Pipeline
            filters (list[Filter]): List of Filters for pipeline
            datasets (list[Dataset]): List of Datasets for pipeline
            scope (str): Definition of scope of the pipeline
            scope_description (str): Description of scope of the pipeline
            summary (Summary, optional): Summary for pipeline. Defaults to None.
            visualizations (list[Visualization], optional): Visualizations for pipeline. Defaults to None.
        """
        super().__init__(pipeline_name, fields, queries, csvs, datasets, scope, scope_description, summary, visualizations)
        self.filters = filters
        
    def build_pipeline_schema(self):
        """Builds Dashboard Pipeline Schema
        """
        super().build_pipeline_schema()
        super().build_filters_schema()