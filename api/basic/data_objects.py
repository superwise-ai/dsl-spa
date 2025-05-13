from pydantic import BaseModel, Field

class PipelineDefinition(BaseModel):
    pipeline_type: str = Field(title = "Pipeline Type", description="Type of Pipeline to create for this schema.", default="standard")
    pipeline_fields: dict = Field(title = "Pipeline Fields", description='Fields to be used by dsl-spa to process the pipeline schema.')
    pipeline_schema: dict = Field(title = "Pipeline Schema", description='Schema for the Pipeline to use to generate all pipeline objects.')

class Filter(BaseModel):
    pipeline_id: int = Field(title = "Pipeline ID", description="ID of Pipeline")
    name: str = Field(title = "Name", description="Name of filter")
    value: str = Field(title = "Value", description="Selected value for filter")