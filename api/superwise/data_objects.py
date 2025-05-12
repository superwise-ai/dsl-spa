from pydantic import BaseModel, Field
from typing import Union

class PipelineRequest(BaseModel):
    prompt: str = Field(title = "Prompt", description="Prompt from user for Agent")

class Pipeline(BaseModel):
    pipeline_type: str = Field(title = "Pipeline Type", description="Type of Pipeline to create for this schema.", default="standard")
    pipeline_fields: dict = Field(title = "Pipeline Fields", description='Fields to be used by dsl-spa to process the pipeline schema.')
    pipeline_schema: dict = Field(title = "Pipeline Schema", description='Schema for the Pipeline to use to generate all pipeline objects.')

class AskPipelineSummary(BaseModel):
    pipeline_id: int = Field(title = "Pipeline ID", description='ID for pipeline to load summary of.')
    query: str = Field(title = "Query", description="Question to ask the pipeline's summary.")
    superwise_app_id: Union[str,None] = Field(title = "Superwise Application ID", description='Application Id to query about the pipeline summary', default=None)

class SelectPipelineVisualization(BaseModel):
    pipeline_id: int = Field(title = "Pipeline ID", description='ID for pipeline to load visualizations of.')
    query: str = Field(title = "Query", description="Question to use to select from pipeline visualizations.")
    superwise_app_id: Union[str,None] = Field(title = "Superwise Application ID", description='Application Id to query with the pipeline visualizations', default=None)

class Filter(BaseModel):
    pipeline_id: int = Field(title = "Pipeline ID", description="ID of Pipeline")
    name: str = Field(title = "Name", description="Name of filter")
    value: str = Field(title = "Value", description="Selected value for filter")