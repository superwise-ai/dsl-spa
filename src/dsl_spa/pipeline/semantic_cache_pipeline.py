from dsl_spa.pipeline.pipeline import BasicPipeline, PipelineException
from dsl_spa.pipeline.pipeline_functions import pipeline_functions_dict
from dsl_spa.pipeline.connector import Connector
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import util
import ast
from typing import Any, Union
import pandas as pd
        
class BasicSemanticCachePipeline(BasicPipeline):
    """The BasicSemanticCachePipeline is a Specialized Pipeline for leveraging Semantic Caches to simplify LLM use cases. This class needs to have its embedding connected in a subclass.
    This class is ideal for implementing simplified Semantic Caches that would otherwise require large semantic caches to cover many specific values.
    """
    def __init__(self,fields_input_dict: dict, json_schema: dict, connectors: dict[str,Connector], field_cache_dictionary: dict[str,Any], functions: dict = pipeline_functions_dict):
        """Creates a BasicSemanticCachePipeline

        Args:
            ields_input_dict (dict): Fields Input defining the fields for the pipeline
            json_schema (dict): The dictionary of the json schema defining the pipeline
            connectors (dict[str,Connector]): List of connectors
            functions (dict, optional): Dictionary mapping all function names to their functions. Defaults to pipeline_functions_dict.
            field_cache_dictionary (dict[str,Any]): Dictionary mapping a DSL-SPA field (in the fields input dictionary) to the generic value used in the semantic cache.
        """
        super().__init__(fields_input_dict, json_schema, connectors, functions)
        self.field_cache_dictionary = field_cache_dictionary
        
    def initialize_data(self):
        self.load_cache()
        
    def process_data(self):
        self.prepare_cache()
        
    def load_cache(self) -> None:
        """Loads the semantic cache
        """
        self.run_queries()
    
    def prepare_cache(self) -> None:
        self.build_datasets()
        
    def _keep_cache_value(self, required_fields: str) -> bool:
        """Determines if a cache value has all the required fields from a comma-separated string of field names

        Args:
            required_fields (str): Comma-separated string of field names where the absence of any of them indicates a cache value should be excluded

        Returns:
            bool: Whether to keep the cache value
        """
        required_fields_list = required_fields.split(',')
        for required_field in required_fields_list:
            if not self.check_for_field(required_field):
                return False
        return True
        
    def _purge_cache_value(self, exclude_fields: str) -> bool:
        """Determines if a cache value has all the fields to be purged based on a comma-separated string of field names

        Args:
            exclude_fields (str): Comma-separated string of field names where the presence of any of them indicates a cache value should be excluded

        Returns:
            bool: Whether to purge the cache value
        """
        exclude_fields_list = exclude_fields.split(',')
        for required_field in exclude_fields_list:
            if self.check_for_field(required_field):
                return True
        return False
        
    def cleanse_cache(self, df: pd.DataFrame, required_fields_column: str = None, exclude_fields_column: str = None) -> pd.DataFrame:
        """Cleanses cache values that do not have the correct set of required and exclude fields

        Args:
            df (pd.DataFrame): Input cache dataset
            required_fields_column (str, optional): Name of column with comma-seperated string indicating the required fields. If None then does not use this column to cleanse the cache. Defaults to None.
            exclude_fields_column (str, optional): Name of column with comma-seperated string indicating the exclude fields (field which, if present in input fields, would indicate this cache value be excluded). If None then does not use this column to cleanse the cache. Defaults to None.

        Returns:
            pd.DataFrame: Cleansed Cache
        """
        if required_fields_column is not None:
            df["_keep"] = df[required_fields_column].apply(self._keep_cache_value)
        else:
            df["_keep"] = [True]*len(df.index)
        if exclude_fields_column is not None:
            df["_purge"] = df[exclude_fields_column].apply(self._purge_cache_value)
        else:
            df["_purge"] = [False]*len(df.index)
        df = df[df._keep & ~df._purge]
        return df
    
    def cleanse_input(self, input_field_name: str) -> str:
        """Cleanses cache input ahead of embedding comparison

        Args:
            input_field_name (str): Name of field containing the value to be compared to the cache

        Returns:
            str: Cleansed input string
        """
        input_value = self.field_dict[input_field_name]
        for field,value in self.field_cache_dictionary.items():
            if self.check_for_field(field):
                field_value = self.get_field(field)
                input_value= input_value.replace(f'{field_value}', f'{value}')                
        return input_value    
    
    def uncleanse_cache_value(self, cache_value: str) -> str:
        """Converts a cleansed cache string to an uncleansed output based on the values in the field dictionary

        Args:
            cache_value (str): String value from cache

        Returns:
            str: The particularized output from cache based on the values in the field dictionary
        """
        for field,value in self.field_cache_dictionary.items():
            if self.check_for_field(field):
                field_value = self.get_field(field)
                cache_value = cache_value.replace(f'{value}', f'{field_value}')                
        return cache_value
    
    def make_cache_comparisons(self, df: pd.DataFrame, field_name: str, similarity_minimum: float = None) -> pd.DataFrame:
        """Makes comparisons of input value to cache values using an embedding model

        Args:
            df (pd.DataFrame): Input dataframe
            field_name (str): Name of field holding cache value to use for comparison.
            similarity_minimum (float, optional): Minimum similarity value to keep in dataset. If None, does not delete any values. Defaults to None.

        Raises:
            NotImplementedError: Indicates this has not been implemented in the subclass

        Returns:
            pd.DataFrame: Matched cache values
        """
        raise NotImplementedError("This needs to be implemented in a subclass with a connection to an embedding model")
    
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
            should_run = self.validate_function_should_run(process)
            if not should_run:
                continue
            process_type = process["type"]
            if process_type == "query":
                data = self.create_dataset_from_query(process)
            elif process_type == "multiplex_query":
                data = self.create_dataset_from_multiplexed_query(process)
            elif process_type == "filter":
                data = self.apply_filters_to_dataset(data,process)
            elif process_type == "function":
                data = self.apply_function_to_dataset(data, process)
            elif process_type == "cleanse_cache":
                data = self.cleanse_cache(data)
            elif process_type == "make_cache_comparisons":
                params = self.get_function_parameters(data, process)
                data = self.make_cache_comparisons(**params)
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
    
    def get_semantic_cache_result(self, top_n = 1) -> list[dict[str,Any]]:
        """Gets the top_n most similar matches from the semantic cache

        Args:
            top_n (int, optional): _description_. Defaults to 1.

        Raises:
            PipelineException: The name of the dataset for the sematnic cache in the schema is not in the pipeline
            PipelineException: Column for cache result from schema is not in dataset
            PipelineException: No definition of the semantic cache found in the schema

        Returns:
            list[dict[str,Any]]: List of dictionaries mapping output columns to their values from the semantic cache dataset
        """
        if "semantic_cache" in self.schema.keys():
            cache_schema = self.schema["semantic_cache"]
            dataset_name = cache_schema["dataset"]
            results_columns = cache_schema["results_columns"]
            if dataset_name not in self.datasets.keys():
                raise PipelineException(f"Dataset {dataset_name} not found in datasets")
            dataset = self.datasets[dataset_name]
            if len(dataset) <= 0:
                if "emtpy_cache_error_message" in cache_schema:
                    return {
                        "error": cache_schema["emtpy_cache_error_message"]
                    }
                else:
                    return {
                        "error": "Empty cache after filtering"
                    }
                    
            for i,row in dataset.iterrows():
                question = row["question"]
                score = row["similarity_score"]
                print(f"{question} had similarity score of {score}")
            
            results = []
            if len(dataset.index == 1):
                results_dict = {}
                for column in results_columns:
                    if column not in dataset.columns:
                        raise PipelineException(f"Column {column} not found in dataset {dataset_name}")
                    results_dict[column] = dataset[column].values[0]
                return results_dict
            for i in range(top_n):
                if i == len(dataset.index):
                    break
                results_dict = {}
                for column in results_columns:
                    if column not in dataset.columns:
                        raise PipelineException(f"Column {column} not found in dataset {dataset_name}")
                    results_dict[column] = dataset[column].values[i]
                print(f"Output: {results_dict}")
                results.append(results_dict)
            return results
        else:
            raise PipelineException("No semantic cache definition found")
        
class OpenAISemanticCachePipeline(BasicSemanticCachePipeline):
    """The OpenAISemanticCachePipeline implements the BasicSemanticCachePipeline. It uses OpenAi styled (can connect to private models with same connection structure) text embeddings to make comparisons
    between cache values and an input value.
    """
    def make_cache_comparisons(self, df: pd.DataFrame, field_name: str, api_key: str, openai_api_base: str = None, model: str = None, similarity_minimum: float = None) -> pd.DataFrame:
        """Makes comparisons of input value to cache values using OpenAI style embedding model

        Args:
            df (pd.DataFrame): Input dataframe
            field_name (str): Name of field holding cache value to use for comparison.
            api_key (str): API key for OpenAI style embedding model
            openai_api_base (str, optional): Base URL for embedding model. Defaults to None.
            model (str, optional): Name of embedding model to use. Defaults to None.
            similarity_minimum (float, optional): Minimum similarity value to keep in dataset. If None, does not delete any values. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        embedding_model = OpenAIEmbeddings(api_key = api_key, openai_api_base = openai_api_base, model = model)
        user_request = self.cleanse_input(field_name)
        input_embedding = embedding_model.embed_query(text = user_request)
        similiarity_scores = []
        print(f"Cache size before comparison: {len(df.index)}")
        for i,row in df.iterrows():
            embedding = ast.literal_eval(row["embedding"])
            similiarity_scores.append(util.cos_sim(input_embedding, embedding).tolist()[0][0])
        df["similarity_score"] = similiarity_scores
        if similarity_minimum is not None:
            df = df[df["similarity_score"] >= similarity_minimum]
        
        print(f"Cache size after comparison: {len(df.index)}")
        
        return df