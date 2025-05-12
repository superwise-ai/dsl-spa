from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from data_objects import Pipeline, PipelineRequest, AskPipelineSummary, SelectPipelineVisualization, Filter
from api_functions import answer_summary_question, select_visualization_name
from random import randint
from settings import Settings
from dsl_spa.pipeline.connector import Connector, BigQueryConnector, MSSQLConnector, LocalCSVConnector
from dsl_spa.pipeline.pipeline import StandardPipeline, DashboardPipeline, PipelineException
from api_functions import create_pipeline_from_schema
import json
from typing import Callable

from superwise_api.superwise_client import SuperwiseClient

# Load Environment Variables
settings = Settings()

# Initialize Bearer Token Authentication Scheme
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != settings.BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer Token",
        )
        
def generate_pipeline_id(pipeline):
    pipeline_id =  randint(10000000000,99999999999)
    while pipeline_id in pipelines.keys():
        pipeline_id =  randint(10000000000,99999999999)
    pipelines[pipeline_id] = pipeline
    return pipeline_id

# TODO: Add any custom Connector definitions
def build_connectors(schema) -> list[Connector]:
    connectors = {}
    for connector_schema in schema:
        name = connector_schema["name"]
        connector_type = connector_schema["type"]
        if connector_type == "MSSQL":
            connector = MSSQLConnector(connector_schema['username'],connector_schema["password"],connector_schema["host"],connector_schema["database"])
        elif connector_type == "bigquery":
            if "location" in connector_schema:
                connector = BigQueryConnector(connector_schema["url"],connector_schema["account_type"],connector_schema["project_id"],connector_schema["location"])
            else:
                connector = BigQueryConnector(connector_schema["url"],connector_schema["account_type"],connector_schema["project_id"])
        elif connector_type == "local_csv":
            connector = LocalCSVConnector(connector_schema["file_location"])
        else:
            continue
        connector.connect()
        connectors[name] = connector
    return connectors
    
# TODO: Change Pipeline Type here to meet your needs
pipelines: dict[int,StandardPipeline] = {}

connectors_json = "/app/connectors.json"
with open(connectors_json, 'r') as f:
        connector_schema = json.load(f)
connectors = build_connectors(connector_schema)

if settings.SUPERWISE_CLIENT_ID != "" and settings.SUPERWISE_CLIENT_SECRET != "":
    sw_client = SuperwiseClient()

# TODO: Add Pipeline Names to Pipeline Types
pipeline_type_dict: dict[str,Callable] = {
    "standard": StandardPipeline,
    "dashboard": DashboardPipeline
    # "custom_pipeline": CustomPipeline
}

# TODO: Add any predefined Pipeline Schema here
pipeline_schema_dict: dict[str,dict] = {
    # "pipeline_name": json.loads("./pipelines/pipeline_name_schema.json")
}

# TODO: If using superwise, add application ID for natural language to intermediate language application
nl_to_il_app_id = ""

# TODO: If using superwise for pipeline summary chat, add application ID for summary chat application
summary_chat_app_id = ""

# TODO: If using superwise for selecting visualization by natural language, add application ID for summary chat application
visualization_picker_app_id = ""

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def test():
    return {"message": "API Running"}
    
@app.post("/create_pipeline", dependencies=[Depends(verify_token)])
def create_pipeline(data: Pipeline):
    try:
        pipeline = pipeline_type_dict[data.pipeline_type](data.pipeline_fields, data.pipeline_schema, connectors)
        pipeline_id = generate_pipeline_id(pipeline)
        pipeline.initialize_data()
        pipeline.process_data()
        return {"pipeline_id": pipeline_id}
    except PipelineException as e:
        return {"error": str(e)}

@app.post("/create_pipeline_with_superwise", dependencies=[Depends(verify_token)])
def create_pipeline(data: PipelineRequest):
    try:
        pipeline = create_pipeline_from_schema(nl_to_il_app_id, data.prompt, pipeline_type_dict, pipeline_schema_dict, connectors, sw_client)
        pipeline_id = generate_pipeline_id(pipeline)
        pipeline.initialize_data()
        pipeline.process_data()
        return {"pipeline_id": pipeline_id}
    except PipelineException as e:
        return {"error": str(e)}

@app.post("/update_filter", dependencies=[Depends(verify_token)])
def update_filter(data: Filter):
    if data.pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    try:
        pipeline = pipelines[data.pipeline_id]
        pipeline.update_filter(data.name, data.value)
        pipeline.process_data()
        return {"status": "success"}
    except PipelineException as e:
        return {"status": "failure", "error": str(e)}

@app.get("/get_scope/{pipeline_id}", dependencies=[Depends(verify_token)])
def get_scope(pipeline_id: int):
    if pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[pipeline_id]
    scope = pipeline.get_scope()
    scope_description = pipeline.get_scope_description()
    return {"pipeline_id": pipeline_id, "scope": scope, "scope_description": scope_description}

@app.get("/get_filters/{pipeline_id}", dependencies=[Depends(verify_token)])
def get_filters(pipeline_id: int):
    if pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[pipeline_id]
    filters = pipeline.get_filters()
    return {"pipeline_id": pipeline_id, "filters": filters}

@app.get("/get_summary/{pipeline_id}", dependencies=[Depends(verify_token)])
def get_summary(pipeline_id: int):
    if pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[pipeline_id]
    summary = pipeline.get_summary()
    return {"pipeline_id": pipeline_id, "summary": summary}

@app.get("/get_visualizations/{pipeline_id}", dependencies=[Depends(verify_token)])
def get_visualizations(pipeline_id: int):
    if pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[pipeline_id]
    visualization_dict = pipeline.get_visualizations()
    return {"pipeline_id": pipeline_id, "visualizations": visualization_dict}

@app.delete("/delete_pipeline/{pipeline_id}", dependencies=[Depends(verify_token)])
def delete_pipeline(pipeline_id: int):
    if pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    del pipelines[pipeline_id]
    return {"status": "success"}

@app.delete("/delete_all_pipelines", dependencies=[Depends(verify_token)])
def delete_all_pipelines():
    count = 0
    for pipeline_id in pipelines.keys():
        del pipelines[pipeline_id]
        count += 1
    return {"count": count}

@app.post("/ask_pipeline_summary", dependencies=[Depends(verify_token)])
def query_summary(data: AskPipelineSummary):
    if data.pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[data.pipeline_id]
    summary = pipeline.get_summary()
    swe_app_id = data.superwise_app_id if data.superwise_app_id is not None else summary_chat_app_id
    response = answer_summary_question(summary, data.query, swe_app_id, sw_client)
    return {"answer": response}

@app.post("/select_visualization", dependencies=[Depends(verify_token)])
def select_visualization(data: SelectPipelineVisualization):
    if data.pipeline_id not in pipelines.keys():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline ID not found",
        )
    pipeline = pipelines[data.pipeline_id]
    visualzations = pipeline.get_visualizations()
    swe_app_id = data.superwise_app_id if data.superwise_app_id is not None else visualization_picker_app_id
    visualization = select_visualization_name(visualzations, data.query, swe_app_id, sw_client)
    return {"vega_lite": visualization}