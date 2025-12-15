# DSL-SPA Pipelines

### Pipeline

The generic Pipeline class implements DSL-SPA fields. On instantiation, it populates any default field values, processes categorical fields, and verifies that the pipeline has all required fields. This class can be inherited from to implement anything a user needs their Agent to do based on a structured output from an LLM.

The two implemented pipeline types in DSL-SPA are:

 - [Data-Querying Pipelines](#data-querying-pipelines)
 - [Command Pipelines](#command-pipelines)

## Data-Querying Pipelines

DSL-SPA has the following base classes for creating agentic data-querying pipelines:

 - [BasicPipeline](#basic-pipeline) - designed for custom chat applications that generate text and visual resposnes
 - [BasicSemanticCachePipeline](#basic-semantic-cache-pipeline) - designed for any custom application that is being determined by a user chat request

DSL-SPA has the following implemented classes for creating agentic data-querying pipelines:
 - [StandardPipeline](#standard-pipeline) - designed for custom chat applications that generate text and visual resposnes
 - [DashboardPipeline](#dashboard-pipeline) - designed for custom dashboards that a user can interact with via chat
 - [OpenAISemanticCachePipeline](#openai-semantic-cache-pipeline) - designed for any custom application that is being determined by a user chat request

### Basic Pipeline

This pipeline is a base class for the data-querying capabilities of DSL-SPA. A user should not instantiate a Basic Pipeline but instead a subclass of it. The Basic Pipeline implements the following tools:

 - Queries - Used to query a database with a query string.
 - Datasets - Used to perform data transformations on pandas datasets (typically generate from Queries)
 - Filters - A tool to implement temporary post-query processing on datasets. Filters can be added/removed during a pipelines lifetime without needing to rerun queries. In essence, it lets the user 'filter' the data further without needing to do more prompting.
 - Summaries - Used to create text outputs from datasets. Summaries can then be used with other LLM applications to generate text-response applications.
 - Visualizations - Used to create graphs/visualizations of datasets.

 In order to leverage the Basic Pipeline class, create a Pipeline that extends BasicPipeline and implement the `initialize_data()` and `process_data()` functions.

### Standard Pipeline

The Standard Pipeline functions as the default pipeline for data-querying in DSL-SPA. It leverages the Queries, Datasets, Summaries, and Visualizations in the BasicPipeline to enable DSL-SPA developers to implement agentic tasks that require data stores. It adds 'scope' and 'scope-description' variables that help DSL-SPA developers to communicate a pipelines's data scope to its users. 

A flow running a standard pipeline would look like:

```
# Create the pipeline
pipeline = StandardPipeline(fields_input_dict=<fields coming from LLM>,json_schema=<data schema for pipeline>,connectors=<dict of database connectors>)
# Run the pipeline
pipeline.initialize_data()
pipeline.process_data()
# Get the summary and visualizations for the user/for further LLM processing
summary = pipeline.get_summary()
visualizations = pipeline.get_visualizations()
```

### Dashboard Pipeline

The Dashboard Pipeline is similar to the standard pipeline but is designed around being an agentic dashboard. It utilizes the filters defined in the BasicPipeline to allow for a user to interact with a dashboard without further prompting (think of each filter as a dropdown in a dashboard). 

A flow running a Dashboard Pipeline would look like:

```
# Create the pipeline
pipeline = DashboardPipeline(fields_input_dict=<fields coming from LLM>,json_schema=<data schema for pipeline>,connectors=<dict of database connectors>)
# Run the pipeline
pipeline.initialize_data()
pipeline.process_data()
# Get the summary for the user/for further LLM processing
summary = pipeline.get_summary()
# Use the visualizations and filters to create an interactive dashboard
visualizations = pipeline.get_visualizations()
filters = pipeline.get_filters()
# After a filter (dropdown) is changed on the dashboard
pipeline.update_filter(filter_name=<name of the filter>, value=<value of filter>)
summary = pipeline.get_summary()
visualizations = pipeline.get_visualizations()
filters = pipeline.get_filters()
```

### Basic Semantic Cache Pipeline

The Basic Semantic Cache Pipeline is the superclass for implementing dsl-spa semantic caches. Using DSL-SPA, a developer can create a genericized Semantic Cache that is capable of matching across different variable values. 

For example: if you are connecting to a database that has many different projects, and you want a user to be able to look up the status of a project by its project name, you can create a field in this pipeline for 'project_name' and map that to a value in your semantic cache, such as 'The Project' such that when executing the action from the semantic cache, it always uses the value 'project_name' but only needs to match on a generic version of the user request 'What is the status of The Project'.

The Basic Semantic Cache Pipeline extends the Basic Pipeline giving the user access to all the tools in the Basic Pipeline. It also implements some core functions of using a genericized semantic cache:

 - `cleanse_input()` - Converts the user input to the genericized in preperation for cache comparisons
 - `cleanse_cache()` - Removes rows from the cache that do not match the set of required fields or has an exclude field designating the cache row should be excluded if the field value is populated
 - `get_semantic_cache_result()` - Creates a list of the top N results from the semantic cache

 The function `cleanse_cache()` and `make_cache_comparisons()`, while being built into the BasicSemanticCachePipeline class, are expected to be called as part of the process for building a dataset. This lets the developer have complete control over what data transformations are applied before and after the semantic cache comparisons are made.

In order to leverage the BasicSemanticCachePipeline, a developer needs to implement the `make_cache_comparisons()` function that uses a text embedding to compare semantic similarities between the user input and the cache values

### OpenAI Semantic Cache Pipeline

The OpenAI Semantic Cache Pipeline implements the `make_cache_comparisons()` function for any OpenAI compatible embedding model.

A flow running a OpenAI Semantic Cache Pipeline would look like:

```
# Create the pipeline
pipeline = OpenAISemanticCachePipeline(fields_input_dict=<fields coming from LLM>,json_schema=<data schema for pipeline>,connectors=<dict of database connectors>, field_cache_dictionary=<The dictionary mapping of DSL-SPA fields to their generic value>)
# Run the pipeline
pipeline.initialize_data()
pipeline.process_data()
# Get the results
results = pipeline.get_semantic_cache_result(top_n=<number of top results desired>)
# Do something with the semantic cache results
```

## Command Pipelines

DSL-SPA has two implemented Command Pipeline classes. The [Command Pipeline](#command-pipeline) allows a developer to map user requests to executable actions. The [Console Command Pipeline](#console-command-pipeline) allows a developer to map user requests to output strings that can be input into a console/terminal.

### Command Pipeline

The Command Pipeline lets a developer map DSL-SPA fields to a specific Command (set of Actions) to take. It expects that there is a field that specifies the name of the command (by default expects `command_name`). Using this field, it then selects a set of Actions (python functions) to execute, in order, with fields and parameters as input to those Actions.

A flow running a Command Pipeline would look like:

```
# Create the pipeline
pipeline = CommandPipeline(fields_input_dict=<fields coming from LLM>,json_schema=<data schema for pipeline>,connectors=<dict of database connectors>, functions_dict=<python functions for each Action>)
# Run the pipeline
pipeline.initiate_command_pipeline()
pipeline.process_command()
# Should the commands generate some data result
results = pipeline.get_command_result()
```

### Console Command Pipeline

The Console Command Pipeline lets a developer map DSL-SPA fields to generating of a string that can be used within a console or terminal. It works the same as [Command Pipeline](#command-pipeline) except it adds the `generate_console_command()` function that should be the last action called for the command. This action generates the console/terminal command and then saves it the `output_field` set in the Action schema.