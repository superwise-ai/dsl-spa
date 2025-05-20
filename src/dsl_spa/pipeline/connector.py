import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import URL
from sqlalchemy.engine.url import make_url
import os
import json

class Connector:
    """Base Connector Class. Any connector should inherit from this class or a sub class.
    """
    
    def connect(self):
        """Connects to the external connection
        """
        raise NotImplementedError("This method must be implemented in a subclass")
    
class DatabaseConnector(Connector):
    """Database Connector. Includes a query method to query a database.
    """
    def __init__(self):
        self.engine = None
    
    def query(self, query_string: str) -> pd.DataFrame:
        """Runs the query_string query using self.engine.

        Args:
            query_string (str): Query string to query

        Returns:
            pd.DataFrame: Pandas Dataframe of query data
        """
        raise NotImplementedError("This method must be implemented in a subclass")
    
class MSSQLConnector(DatabaseConnector):
    """Microsoft SQL Database Connector.
    """
    def __init__(self, uid: str, password: str, host: str, database: str, param_dict: dict[str,str] = None):
        """Creates a Microsoft SQL Connector

        Args:
            uid (str): Username
            password (str): Password
            host (str): Database Host Server
            database (str): Database Name
            param_dict (dict[str,str]): Dictionary of other connection parameters for sqlalchemy to connect to MSSQL DB like driver, encrypt, TrustServerCertificate.
        """
        super().__init__()
        self.uid = uid
        self.password = password
        self.host = host
        self.database = database
        self.param_dict = param_dict
        
    def connect(self):
        """Connects to the external connection
        """
        if self.param_dict is not None:
            connection_url = URL.create(
                "mssql+pyodbc",
                username=self.uid,
                password=self.password,
                host=self.host,
                database=self.database,
                query=self.param_dict
            )
        else:
            connection_url = URL.create(
                "mssql+pyodbc",
                username=self.uid,
                password=self.password,
                host=self.host,
                database=self.database
            )
        self.engine = sa.create_engine(connection_url)
        
    def query(self, sql_query: str):
        return pd.read_sql_query(sql_query, self.engine)
    
class BigQueryConnector(DatabaseConnector):
    """Big Query Database Connector.
    """
    
    def __init__(self, url: str, account_type: str, project_id: str, location: str = None):
        """Creates a Big Query Connector

        Args:
            url (str): GCP connection URL
            account_type (str): GCP account type
            project_id (str): GCP Project ID
            location (str, optional): GCP Server location. Defaults to None.
        """
        super().__init__()
        self.url = url
        self.location = location
        self.account_type = account_type
        self.project_id = project_id
        
    def connect(self):
        """Connects to the external connection
        """
        credentials_dict = {
            "type": self.account_type,
            "project_id": self.project_id
        }
        credentials_info = None
        if "credentials_base64" in make_url(self.url).query:
            credentials_info = json.dumps(credentials_dict)

        if self.location is None:
            self.engine = sa.create_engine(self.url, credentials_info=credentials_info)
        else:
            self.engine = sa.create_engine(self.url, location=self.location, credentials_info=credentials_info)
        
    def query(self, sql_query: str):
        """Runs the query_string query using self.engine.

        Args:
            query_string (str): Query string to query

        Returns:
            pd.DataFrame: Pandas Dataframe of query data
        """
        return pd.read_sql_query(sql_query, self.engine)
    
class LocalCSVConnector(Connector):
    
    def __init__(self, folder: str):
        """Creates a local CSV connector given the folder location of the CSVs.

        Args:
            folder (str): Location of CSV directory
        """
        self.folder = folder
        self.df = None
        
    def connect(self):
        """Does nothing since the CSV is local.
        """
        pass
    
    def query(self, csv_name: str):
        """Runs the query_string query using self.engine.

        Args:
            query_string (str): Query string to query

        Returns:
            pd.DataFrame: Pandas Dataframe of query data
        """
        file_location = os.path.join(self.folder,csv_name)
        return pd.read_csv(file_location)