import pandas as pd
from dsl_spa.pipeline.pipeline_functions import pipeline_functions_dict
from dsl_spa.pipeline.connector import Connector
import altair as alt
from typing import Any

class PipelineException(Exception):
    """Pipeline Exception. A PipelineException is created when an error is generated based on the provided schema.
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
        return f"Pipeline Exception: {self.pipeline_error}"

class Pipeline:
    """The simplest Pipelines. This pipeline implements fields as defined in schema. 
    It supports default values, categorical values, implements other necessary functions for utilizing fields.
    """
    
    def __init__(self,fields_input_dict: dict, json_schema: dict, connectors: list[Connector]):
        """Creates a Pipeline

        Args:
            fields_input_dict (dict): Fields Input defining the fields for the pipeline
            json_schema (dict): The dictionary of the json schema defining the pipeline
            connectors (list[Connector]): List of connectors
        """
        self.pipeline_type = json_schema["pipeline_name"]
        self.field_dict = fields_input_dict
        self.schema = json_schema
        self.connectors = connectors
        self.populate_default_values()
        self.process_categorical_values()
        required_fields = self.build_required_fields_list(self.schema["fields"])
        self.check_for_required_fields(required_fields)
        
    def process_categorical_values(self):
        """Reads the fields in the pipeline schema and converts any categorical field to its appropriate new field.
        """
        self.fill_categorical_values(self.schema["fields"])

    def fill_categorical_values(self,fields_dict: dict, root: list = []):
        """Fills categorical values 

        Args:
            fields_dict (dict): Field Dictionary from Schema
            root (list, optional): List of field sections for this sub-section of fields. Defaults to [].
        """
        for key in fields_dict:
            if isinstance(fields_dict[key], dict):
                self.fill_categorical_values(fields_dict[key],root.append(key))
            else:
                for d in fields_dict[key]:
                    field = ".".join(root+[key,d["name"]])
                    if d["type"] == "categorical" and self.check_for_field(field):
                        new_field = field+"_"+self.get_field(field)
                        self.set_field(new_field,True)

    def populate_default_values(self):
        """Populates all fields with default values that have not been populated by the fields input to their respective default values.
        """
        default_dict = Pipeline.get_default_values(self.schema["fields"])
        self.add_default_values(default_dict)

    def add_default_values(self,default_dict: dict, root: list = []):
        """Given a dictionary of default values, populates the pipelines' field_dict with those values

        Args:
            default_dict (dict): Dictionary of default values to add to pipelines' fields.
            root (list, optional): List of field sections for this sub-section of fields. Defaults to [].
        """
        for key in default_dict.keys():
            if isinstance(default_dict[key], dict):
                self.add_default_values(default_dict[key],root + [key])
            else:
                d = self.field_dict
                for k in root:
                    # Only set field if not already populated
                    if k not in d.keys():
                        d[k] = {}
                    d = d[k]
                # Only set field if not already populated
                if key not in d.keys():
                    d[key] = default_dict[key]

    def get_default_values(field_dict: dict) -> dict:
        """Retrieves all the default values from the schema.

        Args:
            field_dict (dict): Dictionary of fields.

        Returns:
            dict: Default values for each field based on the schema.
        """
        default_dict = {}
        for key in field_dict.keys():
            if isinstance(field_dict[key], list):
                for d in field_dict[key]:
                    if "default" in d.keys():
                        if key == "base":
                            default_dict[d["name"]] = d["default"]
                        else:
                            if key not in default_dict.keys():
                                default_dict[key] = {}
                            default_dict[key][d["name"]] = d["default"]
            else:
                default_dict[key] = Pipeline.get_default_values(field_dict[key])
        return default_dict
    
    def build_required_fields_list(self,fields_dict: dict, root: str = "") -> list:
        """Builds the list of required fields for a pipeline to run.

        Args:
            fields_dict (dict): Dictionary of pipeline fields
            root (str, optional): Root of the current fields_dict. Defaults to "".

        Returns:
            list: List of fields required for Pipeline to run
        """
        if "required" in fields_dict.keys():
            required = fields_dict["required"]
            name = fields_dict["name"]
            if required:
                print(f"adding {root}.{name} as required field")
                return [f"{root}.{name}"]
            else:
                return []
        fields_list = []
        for key in fields_dict.keys():
            if isinstance(fields_dict[key],list):
                for d in fields_dict[key]:
                    if root == "":
                        fields_list.extend(self.build_required_fields_list(d,root=f"{key}"))
                    else:
                        fields_list.extend(self.build_required_fields_list(d,root=f"{root}.{key}"))
                        
            else:
                fields_list.extend(self.build_required_fields_list(fields_dict[key]),root=f"{root}.{key}")
        return fields_list
                
    def check_for_required_fields(self,required_fields: list):
        """Checks that all required fields are included in the fields_dict.

        Args:
            required_fields (list): List of required fields for Pipeline to run

        Raises:
            PipelineException: Error indicating the missing field
        """
        for f in required_fields:
            if not self.check_for_field(f):
                raise PipelineException(f"Required field {f} not found. Make sure to include it in your request.")

    def check_for_field(self, field: str) -> bool:
        """Checks if the field is in the pipeline's field_dict

        Args:
            field (str): Name of field to check for.

        Returns:
            bool: Whether the field was found in the pipeline's field_dict
        """
        field_split = field.split('.')
        root = self.field_dict
        for field_name in field_split:
            if field_name == "base":
                continue
            elif field_name not in root.keys():
                return False
            root = root[field_name]
        return root is not None
    
    def get_field(self, field: str) -> Any:
        """Gets the value of field from the field_dict.

        Args:
            field (str): Name of the field.

        Returns:
            Any: Value of the field.
        """
        field_split = field.split('.')
        value = self.field_dict
        for field_name in field_split:
            if field_name == "base":
                continue
            elif field_name not in value.keys():
                return False
            value = value[field_name]
        return value
    
    def set_field(self, field: str, value: Any):
        """Sets the value of field to value.

        Args:
            field (str): Field name.
            value (Any): Value of the field
        """
        field_split = field.split('.')
        d = self.field_dict
        for k in field_split[:-1]:
            if k == "base":
                continue
            elif k not in d.keys():
                d[k] = {}
            d = d[k]
        d[field_split[-1]] = value

class BasicPipeline(Pipeline):
    """The BasicPipeline utilizes the Fields implemented in Pipeline and also implements queries, filters, datasets, dataset summarization, and visualizations.
    This Pipeline can be used as is, or implemented in a sub-class to generate streamlined pipelines for various use cases.
    """
    def __init__(self,fields_input_dict: dict, json_schema: dict, connectors: list[Connector], functions: dict = pipeline_functions_dict):
        """Creates a BasicPipeline

        Args:
            fields_input_dict (dict): Fields Input defining the fields for the pipeline
            json_schema (dict): The dictionary of the json schema defining the pipeline
            connectors (list[Connector]): List of connectors
            functions (dict, optional): Dictionary mapping all function names to their functions. Defaults to pipeline_functions_dict.
        """
        super().__init__(fields_input_dict, json_schema, connectors)
        self.queries = {}
        self.datasets = {}
        self.dataset_summary_clauses = {}
        self.dataset_summary_prefixes = {}
        self.dataset_summary_suffixes = {}
        self.dataset_summary_remove_commas = {}
        self.functions = functions
        self.visualizations = {}
    
    def add_fields_to_clause(self,clause: str, sanitize_for_sql: bool = False) -> str:
        """Replaces all instances of {field_name} in clause.

        Args:
            clause (str): Clause to replace field values in.
            sanitize_for_sql (bool, optional): Whether to sanitize this clause for sql (convert ' to \\'). Defaults to False.

        Returns:
            str: Clause with fields replaced with their value.
        """
        while '{' in clause:
            start = clause.find('{')
            end = clause.find('}')
            field = clause[start+1:end]
            value = self.get_field(field)
            if sanitize_for_sql and isinstance(value,str):
                value = self.sanitize_field_for_sql_query(value)
            clause = clause[:start] + str(value) + clause[end+1:]
        return clause
    
    def has_required_fields(self,required_fields: list[list[str]]) -> bool:
        """Checks if pipeline has any of the required set of fields.

        Args:
            required_fields (list[list[str]]): A 2D list using an [[A AND B] OR [C AND D]] structure

        Returns:
            bool: Whether the pipeline has the required set of fields.
        """
        for field_list in required_fields:
            has_required = True
            for field in field_list:
                if not self.check_for_field(field):
                    has_required = False
                    print(f"field {field} not found")
                    break
            if has_required:
                print(field_list)
                return True
        return False
        
    def add_columns_to_clause(self, clause: str, row: pd.Series) -> str:
        """Replaces all instances of {column_name} (from row) into clause.

        Args:
            clause (str): Clause to replaces values in.
            row (pd.Series): Row to use for replacing values in the clause.

        Returns:
            str: Clause with values replaced.
        """
        while '{' in clause:
            start = clause.find('{')
            end = clause.find('}')
            column = clause[start+1:end]
            value = row[column]
            clause = clause[:start] + str(value) + clause[end+1:]
        return clause
    
    def sanitize_field_for_sql_query(self,field_value: Any) -> Any:
        """Sanitizes field_value for sql (converts ' to \\')

        Args:
            field_value (Any): Value to sanitize

        Returns:
            Any: Field Value with sanitized input for sql
        """
        field_value = field_value.replace("'","\\'")
        return field_value
    
    def check_query_for_required_fields(self, query: dict) -> bool:
        """Checks if the required fields in the query are in the pipeline's fields

        Args:
            query (dict): The schema dictionary of the query

        Returns:
            bool: Whether the fields required for the query are in the pipeline
        """
        if "required_fields" in query.keys():
            for field in query["required_fields"]:
                if not self.check_for_field(field):
                    return False
        return True
    
    def build_query(self, query: dict) -> str:
        """Contructs a query using its query dict definition from the schema

        Args:
            query (dict): The query dict definition from the schema

        Returns:
            str: Full query for sql
        """
        sql_query = ""
        for clause in query["sql_clauses"]:
            optional = clause["optional"]
            sql = clause["sql"]
            sql = self.add_fields_to_clause(sql,sanitize_for_sql=True)
            if optional and self.check_for_field(clause["field"]):
                sql_query += sql
            elif not optional:
                sql_query += sql
        return sql_query
    
    def check_if_query_has_minimum_number_of_results(self, query: dict):
        """_summary_

        Args:
            query (dict): Query dict definition from the schema

        Raises:
            PipelineException: Error message as defined by the schema
        """
        query_name = query["name"]
        if "min_results" in query.keys() and len(self.queries[query_name]) < query["min_results"]:
            if "error" in query.keys():
                error_message = self.add_fields_to_clause(query["error"])
            else:
                error_message = f"Issue loading query {query_name}."
            raise PipelineException(error_message)
        
    def validate_query_input(self, query: dict) -> bool:
        """Validates the query has the required inputs to be run

        Args:
            query (dict): Query dict definition from the schema

        Returns:
            bool: Whether the necssary data is in the pipeline to be run
        """
        has_required_fields = self.check_query_for_required_fields(query)
        return has_required_fields
        
    def run_query(self, query: dict, sql_query: str):
        """Runs the sql_query based on the connection defined in the query schema.

        Args:
            query (dict): Query dict definition from the schema
            sql_query (str): SQL query to be run
        """
        query_name = query["name"]
        connector_name = query["connector"]
        self.queries[query_name] = self.connectors[connector_name].query(sql_query)
        
    def validate_query_results(self, query: dict):
        """Validates the query output meets the requirements set by the schema

        Args:
            query (dict): Query dict definition from the schema
        """
        self.check_if_query_has_minimum_number_of_results(query)        

    def run_queries(self):
        """Runs all queries in the query schema
        """
        if "queries" in self.schema.keys():
            for query in self.schema["queries"]:
                query_validated = self.validate_query_input(query)
                if query_validated:
                    query_name = query["name"]
                    sql_query = self.build_query(query)
                    self.run_query(query, sql_query)
                    self.validate_query_results(query)
                    self.log.append(f"## Running {query_name}")
                    self.log.append(f"{sql_query}")
                    
    def load_csvs(self):
        """Loads all the CSVs as defined in the csv schema
        """
        if "csvs" in self.schema.keys():
            for csv in self.schema["csvs"]:
                csv_name = csv["name"]
                connector = csv["connector"]
                filename = csv["csv_name"]
                data = self.connectors[connector].query(filename)
                if "column_filters" in csv:
                    for column_filter in csv["column_filters"]:
                        field = column_filter["field"]
                        if self.check_for_field(field):
                            column = column_filter["column"]
                            value = self.add_fields_to_clause(column_filter["value"])
                            data = data[data[column].astype(str) == value]
                self.queries[csv_name] = data

    def load_filters(self):
        """Populates the filters based on the filters schema
        """
        if "filters" in self.schema.keys():
            filters = []
            for f in self.schema["filters"]:
                column_name = f["column_name"]
                query_name = f["query"]
                values = list(filter(lambda x: x is not None,self.queries[query_name][column_name].unique()))
                if len(values) < 2:
                    continue
                if "field" not in f.keys() or not self.check_for_field(f["field"]):
                    filters.append(f)
                    name = f["name"]
                    print(f"Adding filter {name}")
            if len(filters) <= 0:
                return
            for i in range(len(filters)):
                data_filter = filters[i]
                name = data_filter["name"]
                display_name = data_filter["display_name"]
                column_name = data_filter["column_name"]
                query_name = data_filter["query"]
                values = list(filter(lambda x: x is not None,self.queries[query_name][column_name].unique()))
                include_any = data_filter["include_any"]
                if include_any:
                    values = ['Any'] + values
                self.filters[name] = {
                    "display_name": display_name,
                    "column_name": column_name,
                    "values": values,
                    "include_any": include_any,
                    "selected_value": "Any"
                }
        
    def update_filter(self, filter_name: str, value: Any):
        """Updates a filter_name to value

        Args:
            filter_name (str): Name of filter
            value (Any): Value to set filter_name to
        """
        self.filters[filter_name]["selected_value"] = value
        
    def check_dataset_for_required_fields(self, dataset: dict) -> bool:
        """Checks if the dataset has required fields to be created.

        Args:
            dataset (dict): Dataset dict definition from the schema

        Returns:
            bool: Whether the dataset has the required fields
        """
        if "required_fields" in dataset.keys():
            if not self.has_required_fields(dataset["required_fields"]):
                return False
        return True
    
    def check_dataset_for_exclude_fields(self, dataset: dict) -> bool:
        """Checks dataset if it has any of the fields that indicate it should not be run

        Args:
            dataset (dict): Dataset dict definition from the schema

        Returns:
            bool: Whether the dataset doesn't have any exclude fields
        """
        if "exclude_fields" in dataset.keys():
            if self.has_required_fields(dataset["exclude_fields"]):
                return True
        return False
    
    def load_summary_data_from_dataset(self, dataset: dict):
        """Loads the summary definitions from the dataset schema.

        Args:
            dataset (dict): Dataset dict definition from the schema
        """
        dataset_name = dataset["name"]
        if "summarize" in dataset.keys():
            self.dataset_summary_clauses[dataset_name] = dataset["summarize"]
        if "prefix" in dataset.keys():
            self.dataset_summary_prefixes[dataset_name] = dataset["prefix"]
        if "suffix" in dataset.keys():
            self.dataset_summary_suffixes[dataset_name] = dataset["suffix"]
        if "remove_comma" in dataset.keys():
            self.dataset_summary_remove_commas[dataset_name] = dataset["remove_comma"]
    
    def create_dataset_from_query(self, process: dict) -> pd.DataFrame:
        """Creates dataset from query

        Args:
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Results from the query
        """
        query_name = process["name"]
        data = self.queries[query_name]
        self.log.append(f"Loading data from query: {query_name}")
        return data
    
    def create_dataset_from_multiplexed_query(self, process: dict) -> pd.DataFrame:
        """Creates dataset from a set of queries based on a multiplexing field

        Args:
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Results from the query
        """
        for k,v in process["options"].items():
            if self.check_for_field(k):
                data = self.queries[v]
                query_name = v
                break
        self.log.append(f"Loading data from multiplexed query: {query_name}")
        return data
    
    def create_dataset_from_dataset(self, process: dict) -> pd.DataFrame:
        """Creates dataset from another dataset

        Args:
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Dataset from other dataset
        """
        previous_dataset_name = process["name"]
        data = self.datasets[previous_dataset_name]
        self.log.append(f"Loading data from dataset: {previous_dataset_name}")
        return data
    
    def create_dataset_from_merge(self, process: dict) -> pd.DataFrame:
        """Create dataset by merging two other datasets

        Args:
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Dataset from merging two datasets
        """
        dataset1_name = process["dataset1"]
        dataset2_name = process["dataset2"]
        df1 = self.datasets[dataset1_name]
        df2 = self.datasets[dataset2_name]
        data = pd.merge(df1, df2, how=process["how"], left_on=process["left_on"], right_on=process["right_on"])
        if "nan_replace" in process.keys():
            data = data.fillna(process["nan_replace"])
        self.log.append(f"Mergine Datasets {dataset1_name} and {dataset2_name}")
        return data
    
    def apply_filters_to_dataset(self, data: pd.DataFrame, process: dict) -> pd.DataFrame:
        """Applies filters to dataset

        Args:
            data (pd.DataFrame): Dataset to apply filters to
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Dataset with filters applied to it
        """
        filters_to_apply = process["filters"]
        for filter_name in filters_to_apply:
            if filter_name not in self.filters.keys():
                continue
            data_filter = self.filters[filter_name]
            column_name = data_filter["column_name"]
            selected_value = data_filter["selected_value"]
            if not data_filter["include_any"] or selected_value != "Any":
                if column_name in data.columns:
                    unique_column_values = data[column_name].unique()
                    data = data[data[column_name] == selected_value]
                    self.log.append(f"Applying filter {selected_value} on {column_name}")
        return data
    
    def apply_function_to_dataset(self, data: pd.DataFrame, process: dict) -> pd.DataFrame:
        """Applies function with parameters and fields as defined in schema to dataset

        Args:
            data (pd.DataFrame): Dataset to apply function to
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Dataset with function applied to it
        """
        check = True
        if "required_fields" in process.keys():
            for field_name in process["required_fields"]:
                check = check and self.check_for_field(field_name)
        if check:
            function_name = process["name"]
            func = self.functions[function_name]
            params = {
                "df": data
            }
            if "params" in process.keys():
                for k,v in process["params"].items():
                    if isinstance(v,str):
                        v = self.add_fields_to_clause(v)
                    params[k] = v
            if "fields" in process.keys():
                for k,v in process["fields"].items():
                    if self.check_for_field(v):
                        params[k] = self.get_field(v)
                    else:
                        print(f"{v} field not found")
            data = func(**params)
            self.log.append(f"Running function: {function_name}")
        return data
    
    def apply_arithmetic_operation_to_dataset(self, data: pd.DataFrame, process: dict) -> pd.DataFrame:
        """Applies arithmetic operation with parameters as defined in schema to dataset

        Args:
            data (pd.DataFrame): Dataset to apply function to
            process (dict): Specific dataset dict definition from the schema

        Returns:
            pd.DataFrame: Dataset with arithmetic operation applied to it
        """
        operation = process["operation"]
        column = process["column"]
        by = process['by']
        if operation == '+':
            data[column] = data[column] + float(by)
        elif operation == '-':
            data[column] = data[column] - float(by)
        elif operation == '*':
            data[column] = data[column] * float(by)
        elif operation == '/':
            data[column] = data[column] / float(by)
        self.log.append(f"Applying arithmetic operation: {operation}{by} to {column}")
        return data
    
    def create_dataset(self, dataset: dict) -> pd.DataFrame:
        """Creates dataset 

        Args:
            dataset (dict): Dataset dict definition from the schema

        Raises:
            Exception: No dataset was generated based on the schema

        Returns:
            pd.DataFrame: Dataset
        """
        data = None
        for process in dataset["create"]:
            process_type = process["type"]
            if process_type == "query":
                data = self.create_dataset_from_query(process)
            elif process_type == "multiplex_query":
                data = self.create_dataset_from_multiplexed_query(process)
            elif process_type == "filter":
                data = self.apply_filters_to_dataset(data,process)
            elif process_type == "function":
                data = self.apply_function_to_dataset(data, process)
            elif process_type == "merge":
                data = self.create_dataset_from_merge(process)
            elif process_type == "dataset":
                data = self.create_dataset_from_dataset(process)
            elif process_type == "arithmetic":
                data = self.apply_arithmetic_operation_to_dataset(data, process)
        if data is None:
            dataset_name = dataset["name"]
            raise Exception(f"Dataset {dataset_name} is missing necessary operations to create.")
        return data
    
    def build_dataset(self, dataset: dict):
        """Builds dataset and any summary definitions from dataset schema

        Args:
            dataset (dict): Dataset dict definition from the schema
        """
        if not self.check_dataset_for_required_fields(dataset):
            return
        elif self.check_dataset_for_exclude_fields(dataset):
            return
        
        dataset_name = dataset["name"]
        data = None
        
        self.load_summary_data_from_dataset(dataset)

        self.log.append(f"## Building {dataset_name}")

        data = self.create_dataset(dataset)
        self.datasets[dataset_name] = data

    def build_datasets(self):
        """Builds all datasets in the dataset schema
        """
        if "datasets" in self.schema.keys():
            for dataset in self.schema["datasets"]:
                self.build_dataset(dataset)
                                        
    def get_datasets(self) -> dict:
        """Gets all datasets

        Returns:
            dict: Dictionary of dataset names and their pandas dataframes
        """
        return self.datasets
        
    def summarize_dataset(self, dataset_name: str) -> str:
        """Generates dataset summary given a dataset name

        Args:
            dataset_name (str): Name of Dataset

        Returns:
            str: Summary of dataset
        """
        dataset = self.datasets[dataset_name]
        summary_clause = self.dataset_summary_clauses[dataset_name]        
        summary = ""
        
        if dataset_name in self.dataset_summary_prefixes.keys():
            summary += self.add_fields_to_clause(self.dataset_summary_prefixes[dataset_name])
        
        for i,row in dataset.iterrows():
            clause = self.add_columns_to_clause(summary_clause,row)
            summary += clause
        
        if dataset_name in self.dataset_summary_remove_commas.keys() and self.dataset_summary_remove_commas[dataset_name]:
            if summary[-1] == ',':
                summary = summary[:-1]
            elif summary[-2:] == ', ':
                summary = summary[:-2]
        
        if dataset_name in self.dataset_summary_suffixes.keys():
            summary += self.add_fields_to_clause(self.dataset_summary_suffixes[dataset_name])
            
        return summary
    
    def build_summary(self):
        """Builds summary from summary of all datasets in the summary schema
        """
        if "summary" not in self.schema.keys():
            self.summary = None
            return
        
        summary = ""
        if "prefix" in self.schema["summary"].keys():
            summary = self.add_fields_to_clause(self.schema["summary"]["prefix"])
        datasets_to_summarize = self.schema["summary"]["datasets"]
        for dataset_name in datasets_to_summarize:
            summary += self.summarize_dataset(dataset_name)
        if "suffix" in self.schema["summary"].keys():
            summary = self.add_fields_to_clause(self.schema["summary"]["suffix"])
        self.summary = summary
    
    def get_summary(self) -> str:
        """Gets the summary of the pipeline

        Returns:
            str: Summary of the pipeline
        """
        if self.summary is None:
            return ""
        return self.summary
    
    def add_variables_to_graph_titles(self,visualization: dict):
        """Adds any field names to graph titles

        Args:
            visualization (dict): Single visualization dict from schema
        """
        if visualization["type"] in ["line","pie","stacked_bar","histogram"]:
            visualization["title"] = self.add_fields_to_clause(visualization["title"])
        elif visualization["type"] == "split_graph":
            for v in visualization["graphs"]:
                v["title"] = self.add_fields_to_clause(v["title"])
        
    def get_visualizations(self) -> dict:
        """Gets all visualizations as dict mapping visualization name to visualization vega-lite schema

        Returns:
            dict: Visualizations
        """
        if self.visualizations is None:
            return {}
        return self.visualizations
    
    def build_visualizations(self):
        """Builds all visualizations as defined in the schema
        """
        if "visualizations" not in self.schema.keys():
            self.visualizations = None
            return
        visualizations = self.schema["visualizations"]
        for v in visualizations:
            self.add_variables_to_graph_titles(v)
            vega_lite_dict = self.get_visualization_dict()
            title = v["title"]
            self.visualizations[title] = vega_lite_dict            
    
    def get_visualization_dict(self, visualization: dict, dataset: pd.DataFrame) -> dict:
        """Gets vega-lite dict given visualization dict and dataset for visualization

        Args:
            visualization (dict): Visualization dict
            dataset (pd.DataFrame): Dataset to build visualization with

        Raises:
            ValueError: Visualization type is not supported

        Returns:
            dict: Vega-lite schema dict
        """
        chart = None
        visualization_type = visualization["type"]
        if visualization_type == "line":
            chart = self.draw_line_graph(visualization,dataset)
        elif visualization_type == "pie":
            chart = self.draw_pie_graph(visualization,dataset)
        elif visualization_type == "histogram":
            chart = self.draw_histogram(visualization,dataset)
        elif visualization_type == "stacked_bar":
            chart = self.draw_stacked_bar_chart(visualization,dataset)
        if chart is None:
            raise ValueError(f"Visualization Type {visualization_type} is not supported.")
        return chart.to_dict()
    
    def draw_line_graph(self, visualization_dict: dict, dataset: pd.DataFrame) -> alt.Chart:
        """Draws line graph

        Args:
            visualization (dict): Visualization dict
            dataset (pd.DataFrame): Dataset to build visualization with
            
        Returns:
            alt.Chart: Altair Line Graph
        """
        x_axis = visualization_dict["x_axis"]
        y_axis = visualization_dict["y_axis"]
        title = visualization_dict["title"]
        if len(dataset.index) > 0:
            if isinstance(y_axis, str):
                chart = alt.Chart(dataset, title=title).mark_line().encode(x=x_axis, y=y_axis)
                chart = chart.configure_title(orient='top', anchor='middle')
                return chart
            else:
                y_axis_name = visualization_dict["y_axis_name"] if "y_axis_name" in visualization_dict.keys() else "count"
                column = y_axis[0]
                data = dataset[[x_axis] + [column]]
                num_rows = len(data.index)
                label_values = [column]*num_rows
                data["label"] = label_values
                data = data.rename(columns={x_axis: x_axis, "label": "label", column: y_axis_name})
                for column in y_axis[1:]:
                    partial_data = dataset[[x_axis] + [column]]
                    num_rows = len(data.index)
                    label_values = [column]*num_rows
                    partial_data["label"] = label_values
                    partial_data = partial_data.rename(columns={x_axis: x_axis, "label": "label", column: y_axis_name})
                    data = pd.concat([data,partial_data])
                chart = alt.Chart(data, title=title).mark_line().encode(x=x_axis, y=y_axis_name, color="label")
                chart = chart.configure_title(orient='top', anchor='middle')
                return chart
        else:
            return None

    def draw_pie_graph(self, visualization_dict: dict, dataset: pd.DataFrame) -> alt.Chart:
        """Draws pie graph

        Args:
            visualization (dict): Visualization dict
            dataset (pd.DataFrame): Dataset to build visualization with
            
        Returns:
            alt.Chart: Altair Pie Graph
        """
        graph_title = visualization_dict["title"]
        value_column = visualization_dict["value_column"]
        label_column = visualization_dict["label_column"]
        if len(dataset.index) > 0:
            chart = alt.Chart(dataset, title=graph_title).mark_arc().encode(theta=value_column,color=label_column)
            return chart
        else:
            return None
        
    def draw_histogram(self, visualization_dict: dict, dataset: pd.DataFrame) -> alt.Chart:
        """Draws histogram

        Args:
            visualization (dict): Visualization dict
            dataset (pd.DataFrame): Dataset to build visualization with
            
        Returns:
            alt.Chart: Altair Histogram
        """
        graph_title = visualization_dict["title"]
        value_column = visualization_dict["value_column"]
        if len(dataset.index) > 0:
            partial_data = dataset[[value_column]]
            chart = alt.Chart(partial_data, title=graph_title).mark_bar().encode(alt.X(value_column, bin=True), y='count()')
            return chart
        else:
            return None
        
    def draw_stacked_bar_chart(self, visualization_dict: dict, dataset: pd.DataFrame) -> alt.Chart:
        """Draws stacked bar chart

        Args:
            visualization (dict): Visualization dict
            dataset (pd.DataFrame): Dataset to build visualization with
            
        Returns:
            alt.Chart: Altair Stacked Bar Chart
        """
        graph_title = visualization_dict["title"]
        value_column = visualization_dict["value_column"]
        index_column = visualization_dict["index_column"]
        color_column = visualization_dict["color_column"]
        if len(dataset.index) > 0:
            chart = alt.Chart(dataset, title=graph_title).mark_bar().encode(x=index_column,y=f"sum({value_column})",color=color_column)
            return chart
        else:
            return None

class StandardPipeline(BasicPipeline):
    """The StandardPipeline implements the queries, datasets, dataset summarization, and visualizations from the BasicPipeline in a streamlined format. It also includes a pipeline scope and scope description value.
    """
    def initialize_data(self):
        """Initializes data by running queries and loading CSVs
        """
        self.run_queries()
        self.load_csvs()
        
    def process_data(self):
        """Processes data by building datasets, building the pipeline summary, and building visualizations
        """
        self.build_datasets()
        self.build_summary()
        self.build_visualizations()
        
    def get_scope(self) -> str:
        """Gets the scope of the pipeline

        Returns:
            str: Scope of the pipeline
        """
        if "scope" not in self.schema.keys():
            return ""
        return self.add_fields_to_clause(self.schema["scope"])
    
    def get_scope_description(self) -> str:
        """Gets the scope description

        Returns:
            str: Full description of the scope of the pipeline
        """
        if "scope_description" not in self.schema.keys():
            return ""
        return self.add_fields_to_clause(self.schema["scope_description"])
    
class DashboardPipeline(StandardPipeline):
    """The DashboardPipeline uses the implementation of the queries, datasets, dataset summarization, and visualizations from the Standardipeline and also implements filters.
    This pipeline is ideal for an agentic Dashboard.
    """
    def __init__(self, fields_input_dict, json_schema, connectors, functions = pipeline_functions_dict):
        """Creates a Dashboard Pipeline

        Args:
            fields_input_dict (dict): Fields Input defining the fields for the pipeline
            json_schema (dict): The dictionary of the json schema defining the pipeline
            connectors (list[Connector]): List of connectors
            functions (dict, optional): Dictionary mapping all function names to their functions. Defaults to pipeline_functions_dict.
        """
        super().__init__(fields_input_dict, json_schema, connectors, functions)
        self.filters = {}
    
    def initialize_data(self):
        self.run_queries()
        self.load_csvs()
        self.load_filters()