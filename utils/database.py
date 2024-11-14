# Utility class to get database schema, create tables, and run SQL queries
import sqlite3
import re
from sqlalchemy import create_engine, MetaData, text
import boto3
import json
import time
import pandas as pd

class DatabaseUtil():
    def __init__(self,
        debug: bool = False,
        sql_database: str = 'LOCAL',
        sql_database_name: str = 'dev',
        region: str = 'us-east-1',
        s3_bucketname: str = '',
        secret_name: str = ''

    ):
        self.debug = debug
        self.sql_database = sql_database
        self.sql_database_name = sql_database_name
        self.region = region
        self.s3_bucketname = s3_bucketname
        self.secret_name = secret_name

    # retrieve AWS secret for database connection
    def get_secret(self, secret_name):
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=self.region)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return get_secret_value_response

    def get_table_reflections(self, engine) -> MetaData:
    
        # Instantiate MetaData object
        metadata = MetaData()
        
        # Reflect the database schema with the engine
        metadata.reflect(bind=engine)
        
        return metadata

    def convert_reflection_to_dict(self, metadata: MetaData) -> dict:
        table_definitions: list[dict] = []
        for table_name in metadata.tables:
            definition = {}
            definition['table'] = table_name
            # The metadata.table[x].columns value is type sqlalchemy.sql.base.ReadOnlyColumnCollection
            # Lets convert it into something more usable. c.type returns a SQLAlchemy object so we convert to string.
            definition['columns'] = { c.name: str(c.type) for c in metadata.tables[table_name].columns }
        
            table_header = f"Table: {table_name}"
            columns_definition = '\n'.join([f"Column: {c.name}, Type: {c.type}" for c in metadata.tables[table_name].columns])
            string_representation = f"{table_header}\n{columns_definition}"
        
            definition['string_representation'] = string_representation
        
            table_definitions.append(definition)
        
        
        # The metadata table is a FacadeDict object which is immutable so we need to remove unwanted tables in the new list.
        table_names_to_exclude = set(['table_embedding', 'alembic_version'])
        table_definitions = [d for d in table_definitions if d['table'] not in table_names_to_exclude]

        return table_definitions
    
    def create_database_tables(self, sql_statements):

        if self.sql_database == 'LOCAL':
            try:
                # Create a SQLite database connection
                conn = sqlite3.connect('devdb.db')
                cursor = conn.cursor()

                # Execute each SQL statement
                for statement in sql_statements:
                    # Skip empty statements
                    if statement.strip():
                        # print(f'statement: {statement}')
                        # Replace PostgreSQL-specific syntax with SQLite equivalents
                        statement = statement.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
                        statement = statement.replace('::int', '')
                        statement = statement.replace('::varchar', '')
                        statement = statement.replace('::real', '')
                        statement = statement.replace('::date', '')
                        statement = statement.replace('::boolean', '')
                        statement = statement.replace('public.', '')
                        statement = re.sub(r'WITH \(.*?\)', '', statement)
                        
                        try:
                            cursor.execute(statement)
                        except sqlite3.Error as e:
                            print(f"Error executing statement: {e}")

                # Commit the changes and close the connection
                conn.commit()
                conn.close()

                print("SQL execution completed.")
            except Exception as e:
                print(f"Error creating tables: {e}")
                raise

        if self.sql_database == 'REDSHIFT':
            try:
                rdc = boto3.client('redshift-data')
                get_secret_value_response = self.get_secret(self.secret_name)
                # parse REDSHIFT_CLUSTER_DETAILS to extract WorkgroupName, Database, DbUser
                WorkgroupName = json.loads(get_secret_value_response['SecretString']).get('namespaceName')
                Database = json.loads(get_secret_value_response['SecretString']).get('dbname')
                DbUser = json.loads(get_secret_value_response['SecretString']).get('username')

                for statement in sql_statements:
                    try:        
                        rdc.execute_statement(
                            WorkgroupName=WorkgroupName,
                            Database=Database,
                            # DbUser=DbUser,
                            Sql=statement
                        )
                        
                    except Exception as e:
                        print(f"Error executing statement: {e}")
                print("SQL execution completed.")
            except Exception as e:
                print(f"Error creating tables: {e}")
                raise
        
        if self.sql_database =='SQLALCHEMY':
            # create tables in database
            try:
                # SQLALCHEMY_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{SQL_DATABASE_NAME}"
                get_secret_value_response = self.get_secret(self.secret_name)
                SQLALCHEMY_URL = get_secret_value_response['SecretString']

                engine = create_engine(SQLALCHEMY_URL)
                with engine.connect() as connection:
                    # Execute each SQL statement
                    for statement in sql_statements:
                        # Skip empty statements
                        if statement.strip():
                            # print(f'statement: {statement}')
                            # Replace PostgreSQL-specific syntax with SQLite equivalents
                            statement = statement.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
                            statement = statement.replace('::int', '')
                            statement = statement.replace('::varchar', '')
                            statement = statement.replace('::real', '')
                            statement = statement.replace('::date', '')
                            statement = statement.replace('::boolean', '')
                            statement = statement.replace('public.', '')
                            statement = statement.replace('VARBYTE', 'bytea')
                            statement = statement.replace('bpchar', 'varchar')
                            
                            statement = re.sub(r'WITH \(.*?\)', '', statement)
                            
                            try:
                                connection.execute(text(statement))
                            except Exception as e:
                                print(f"Error executing statement: {e}")
                    connection.commit()
                    print("SQL execution completed.")    
            except Exception as e:
                print(f"Error creating tables: {e}")
                raise


    def get_schema_as_string(self):
        if self.sql_database == 'LOCAL':
            db_path = 'devdb.db'          
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Query to get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema_string = ""

            for table in tables:
                table_name = table[0]
                # Query to get the CREATE TABLE statement for each table
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                create_table_stmt = cursor.fetchone()[0]
                
                schema_string += f"{create_table_stmt};\n\n"

            conn.close()
            return schema_string

        if self.sql_database =='SQLALCHEMY':
            try:
                # Use SQLAlchemy if SQL Alchemy is used
                # SQLALCHEMY_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{SQL_DATABASE_NAME}"
                get_secret_value_response = self.get_secret(self.secret_name)
                SQLALCHEMY_URL = get_secret_value_response['SecretString']
                
                engine = create_engine(SQLALCHEMY_URL)
                metadata = self.get_table_reflections(engine)
                table_definitions = self.convert_reflection_to_dict(metadata)

                return table_definitions
                # with engine.connect() as connection:
                #     result = connection.execute("""""")
                #     return result.fetchall()
            except Exception as e:
                error = f"Error executing statement: {e}"
                print(error)

        if self.sql_database == "REDSHIFT":
            try:
                get_secret_value_response = self.get_secret(self.secret_name)
                # parse REDSHIFT_CLUSTER_DETAILS to extract WorkgroupName, Database, DbUser
                WorkgroupName = json.loads(get_secret_value_response['SecretString']).get('namespaceName')
                Database = json.loads(get_secret_value_response['SecretString']).get('dbname')
                DbUser = json.loads(get_secret_value_response['SecretString']).get('username')
                
                rdc = boto3.client('redshift-data')
                result = rdc.execute_statement(
                    WorkgroupName=WorkgroupName,
                    Database=Database,
                    # DbUser=DbUser,
                    Sql=f"select * from pg_table_def where schemaname = 'public';"
                )
                return result
            except Exception as e:
                print(f"Error executing statement: {e}")
      
            
        if self.sql_database == 'GLUE':
            # use a Glue database
            table_names=None
            try:
                glue_client = boto3.client('glue', region_name=self.region)
                table_schema_list = []
                response = glue_client.get_tables(DatabaseName=self.sql_database_name)

                all_table_names = [table['Name'] for table in response['TableList']]

                if table_names:
                    table_names = [name for name in table_names if name in all_table_names]
                else:
                    table_names = all_table_names

                for table_name in table_names:
                    response = glue_client.get_table(DatabaseName=self.sql_database_name, Name=table_name)
                    columns = response['Table']['StorageDescriptor']['Columns']
                    schema = {column['Name']: column['Type'] for column in columns}
                    table_schema_list.append({"Table: {}".format(table_name): 'Schema: {}'.format(schema)})
            except Exception as e:
                print(f"Error: {str(e)}")
            return table_schema_list
        
    def run_sql(self, statement):
    
        if self.sql_database == 'LOCAL':
            try:
                # Create a SQLite database connection
                conn = sqlite3.connect('devdb.db')
                cursor = conn.cursor()

                cursor.execute(statement)
                # Fetch all rows from the result
                result = cursor.fetchall()
                conn.close()
                return result
            except sqlite3.Error as e:
                error = f"Error executing statement: {e}"
                raise
            
            finally:
                conn.close()
        
        if self.sql_database == "REDSHIFT":
            try:
                get_secret_value_response = self.get_secret(self.secret_name)
                # parse REDSHIFT_CLUSTER_DETAILS to extract WorkgroupName, Database, DbUser
                WorkgroupName = json.loads(get_secret_value_response['SecretString']).get('namespaceName')
                Database = json.loads(get_secret_value_response['SecretString']).get('dbname')
                DbUser = json.loads(get_secret_value_response['SecretString']).get('username')

                rdc = boto3.client('redshift-data')
                result = rdc.execute_statement(
                    WorkgroupName=WorkgroupName,
                    Database=Database,
                    # DbUser=DbUser,
                    Sql=statement
                )
                # Wait for the query to complete
                query_id = result['Id']
                while True:
                    status = rdc.describe_statement(Id=query_id)
                    if status['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                        break
                    time.sleep(0.5)  # Wait for 0.5 seconds before checking again

                if status['Status'] == 'FAILED':
                    raise Exception(f"Query failed: {status.get('Error', 'Unknown error')}")

                # Fetch the results
                response = rdc.get_statement_result(Id=query_id)
                
                # Extract column names and data types
                columns = [col['name'] for col in response['ColumnMetadata']]
                
                # Extract data
                data = []
                for row in response['Records']:
                    data.append([field.get('stringValue') or field.get('longValue') or field.get('doubleValue') for field in row])
                
                
                # Create pandas DataFrame
                df = pd.DataFrame(data, columns=columns)                
                
                return df

                
            except Exception as e:
                print(f"Error executing statement: {e}")

        if self.sql_database =='SQLALCHEMY':
            try:
                # SQLALCHEMY_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{SQL_DATABASE_NAME}"
                get_secret_value_response = self.get_secret(self.secret_name)
                SQLALCHEMY_URL = get_secret_value_response['SecretString']
                
                engine = create_engine(SQLALCHEMY_URL)
                with engine.connect() as connection:
                    result = connection.execute(text(statement))
                    return result.fetchall()
            except Exception as e:
                error = f"Error executing statement: {e}"
                raise