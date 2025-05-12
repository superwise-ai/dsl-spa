import requests
import json
from superwise_api.superwise_client import SuperwiseClient
from dsl_spa.pipeline.connector import Connector
from dsl_spa.pipeline.pipeline_functions import pipeline_functions_dict
from dsl_spa.pipeline.pipeline import StandardPipeline

def ask_swe_application_via_api(sw: SuperwiseClient, app: str, user_input: str):
    endpoint_url = f"https://api.superwise.ai/v1/app-worker/{app}/v1/ask"
    
    token = str(sw.application.get_by_id(_id=app).api_token)
        
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-token": token
    }
    payload = {
        "chat_history": [],
        "input": user_input
    }
    
    resp = requests.post(endpoint_url, json=payload, headers=headers)
    app_response = resp.json()
    return app_response["output"]

def create_pipeline_from_schema(swe_application_id: str, prompt: str, pipeline_type_dict: dict[str,function], pipeline_schema_dict: dict[str,dict], connectors: list[Connector], sw_client: SuperwiseClient, pipeline_functions_dict: dict[str,function] = pipeline_functions_dict) -> StandardPipeline:
    response = ask_swe_application_via_api(sw_client, swe_application_id, prompt)
    if response[0] != "{":
        raise ValueError(f"Superwise Application did not return json: {response}")
    pipeline_fields = json.loads(response)
    pipeline_name = pipeline_fields["pipeline_name"]
    pipeline_schema = pipeline_schema_dict[pipeline_name]
    pipeline_type = pipeline_type_dict[pipeline_name]
    return pipeline_type(pipeline_fields, pipeline_schema, connectors, pipeline_functions_dict)

def answer_summary_question(summary: str, user_request: str, superwise_app_id: str, sw: SuperwiseClient):
    print(f"Querying Summary: {user_request}\n")
    
    swe_app_input = f"""Summary: {summary}
    Question: {user_request}
    """
    project_specific_response = ask_swe_application_via_api(sw,superwise_app_id, swe_app_input)
    
    return project_specific_response

def select_visualization_name(visualizations: dict, user_request: str, superwise_app_id: str, sw: SuperwiseClient):
    swe_app_input = f"""List of Visualizations and their descriptions:
    """
    for title in visualizations.keys():
        description = visualizations[title]["description"]
        swe_app_input += f"{title} - {description}\n"
    swe_app_input += f"Question: {user_request}"
    
    visualization_title = ask_swe_application_via_api(sw,superwise_app_id, swe_app_input)
    
    return visualizations[visualization_title]["vega_lite"]