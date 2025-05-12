# Building Pipeline Agent API

## Necessary Files

Running an API requires a .env and connectors.json files.

### Environment Variables

#### Basic

You will need a bearer token for the API to use for authentication

```
BEARER_TOKEN=<bearer token>
```

### Connectors

You will need a json file defining the connectors your dsl-spa pipelines will use.

```
[
    {
        "name": "<name for csv in pipeline>",
        "file_location": "<data_location>",
        "type": "local_csv"
    },
    {
        "name": "<name for mssql database in pipeline>",
        "type": "MSSQL",
        "username": "username",
        "password": "password",
        "host": "<ip_address>,<port>",
        "database": "<database name>",
        "driver": "<ODBC driver name>"
    },
    {
        "name": "<name for bigquery database in pipeline>",
        "type": "bigquery",
        "url": "bigquery://<location of bigquery instance>",
        "account_type": "<account type, likely service_account>",
        "project_id": "<project id>",
        "location": "<OPTIONAL, google cloud server location of bigquery instance>
    }
]
```

## Building with Docker

### Build Docker Image

```
docker build -t pipeline-api .
```

### Start Docker Container

```
docker run -d --name pipeline-container -p 8000:80 pipeline-api
```

This maps the api to port 8000. You can access the api docs locally at `localhost:8000/docs`.