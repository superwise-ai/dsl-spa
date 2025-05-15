# Domain Specific Language - Structured Pipeline Agent

DSL-SPA is an open-source Python library for connecting LLM Agents to ETL and other common tasks.  With this tool, you can connect LLM Agents to complex Data Pipelines.

Key Features of DSL-SPA

- Building SQL Queries - construct sql queries from fields extracted by Agents

- Applying Data Transformations - selectively apply data transformations based on Agent instructions

- Generating Unstructured Text Summaries - generate summaries of query results and data transformations 

- Generating Visualizations - generate vega-lite visualizations of query results and data transformations

For some examples of what you can do with dsl-spa sdk check out the [streamlit demo](https://pipeline-agent.streamlit.app/).

## Getting Started

Get started with dsl-spa by installing the Python library via pip

```
pip install dsl-spa
```

A standard pipeline can be imported with

```
from dsl_spa.pipeline.pipeline import StandardPipeline
```

For more details on constructing a pipeline, reference [Building a Pipeline](https://github.com/superwise-ai/dsl-spa/blob/main/docs/Creating_a_Pipeline_Schema.md).