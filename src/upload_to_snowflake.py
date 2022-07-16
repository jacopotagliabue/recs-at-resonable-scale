"""

    One-off script to upload the H&M Dataset into a Snowflake warehouse. We leverage here:

    * the dataset wrapper provided by H&M (https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=customers.csv)
    * the data logic explained in our paas ingestion repo (https://github.com/jacopotagliabue/paas-data-ingestion).

"""
import os
import csv
import time
import uuid
import json
from dotenv import load_dotenv
from datetime import datetime
import tempfile
from snowflake_client import SnowflakeClient
# load envs
load_dotenv(verbose=True)


def create_schema(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    ):
    sql_query = "create schema IF NOT EXISTS {}.{};".format(snowflake_db.upper(),
                                                            snowflake_schema.upper())

    return snowflake_client.execute_query(sql_query)


def create_table(
    snowflake_client : SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_table: str
    ):
    """
    
    Original file is from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
    
    """
    # attention, table is RE-CREATED!
    print("====== TABLE IS BEING DROPPED AND RE-CREATED =====")
    sql_query = """
    CREATE OR REPLACE TABLE 
    {}.{}.{}(
        etl_timestamp int,
        etl_id VARCHAR(36),
        event_type string,
        raw_data VARIANT
    );
    """.format(snowflake_db.upper(),
               snowflake_schema.upper(),
               snowflake_table.upper())

    return snowflake_client.execute_query(sql_query, is_debug=True)


def use_database(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str
    ):
    sql_query = "USE DATABASE {};".format(snowflake_db.upper())

    return snowflake_client.execute_query(sql_query)


def stage_data(
    snowflake_client: SnowflakeClient, 
    snowflake_schema: str,
    snowflake_table: str,
    data_file: str,
    data_folder: str
    ):
    sql_query = "PUT file://{}/{} @{}.%{} auto_compress=true overwrite=true".format(
        data_folder,
        data_file,
        snowflake_schema.upper(),
        snowflake_table.upper()
        )

    return snowflake_client.execute_query(sql_query, is_debug=True)


def copy_data(
    snowflake_client: SnowflakeClient, 
    snowflake_schema: str,
    snowflake_table: str,
    data_file: str,
    ):
    sql_query = """
        COPY INTO {}.{} FROM @{}.%{}/{}.gz FILE_FORMAT = (TYPE=CSV, SKIP_HEADER=1, FIELD_OPTIONALLY_ENCLOSED_BY='"')
    """.format(
        snowflake_schema.upper(),
        snowflake_table.upper(),
        snowflake_schema.upper(),
        snowflake_table.upper(),
        data_file
        )
    return snowflake_client.execute_query(sql_query)


def upload_shopping_data(
    snowflake_client: SnowflakeClient, 
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_table: str,
    data_folder: str,
    data_file: str
    ):
    print("Uploading shopping data: {}".format(snowflake_table))
    create_table(snowflake_client, snowflake_db, snowflake_schema, snowflake_table)
    use_database(snowflake_client, snowflake_db)
    stage_data(snowflake_client, snowflake_schema, snowflake_table, data_file, data_folder)
    copy_data(snowflake_client, snowflake_schema, snowflake_table, data_file)
    return


def prepare_shopping_data(
    folder: str,
    table_name: str,
    data_folder: str
):
    print("Preparing data locally: {}".format(table_name))
    data_file_name = table_name + '.csv'
    target_data_file = '{}_dump.csv'.format(table_name)
    etl_timestamp = int(time.time() * 1000)
    etl_id = str(uuid.uuid4()) 
    with open(os.path.join(folder, target_data_file), 'w') as csvfile:
        # NOTE: this list has the same ordering as the CREATE TABLE statement
        fieldnames = ['etl_timestamp', 'etl_id', 'event_type', 'raw_data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        with open(os.path.join(data_folder, data_file_name)) as csvfile:
            reader = csv.DictReader(csvfile)
            for event in reader:
                row = {}
                row['etl_timestamp'] = etl_timestamp
                row['etl_id'] = etl_id
                row['event_type'] = table_name
                row['raw_data'] = json.dumps(event)
                writer.writerow(row)

    return target_data_file


def upload_data_to_snowflake(
    snowflake_client: SnowflakeClient,
    snowflake_db: str,
    snowflake_schema: str,
    snowflake_tables: list, # list of files to upload as tables
    data_folder: str # folder containing the data files 
):
    print('Starting ops at {} '.format(datetime.utcnow()))
    # first, create schema
    create_schema(snowflake_client, snowflake_db, snowflake_schema)
    # create a temp dir to bulk upload into Snowflake
    with tempfile.TemporaryDirectory() as tmpdirname:
        # loop over tables, which should correspond 1:1 to files in the data folder
        for table in snowflake_tables:
            data_file = prepare_shopping_data(tmpdirname, table, data_folder)
            # create table if not there (drop it if there), upload data from CSV dump
            upload_shopping_data(
                snowflake_client,
                snowflake_db=snowflake_db,
                snowflake_schema=snowflake_schema,
                snowflake_table=table,
                data_folder=tmpdirname,
                data_file=data_file
                )

    print('All done, see you, space cowboy {} '.format(datetime.utcnow()))
    return


if __name__ == "__main__":
    # init snowflake client
    sf_client = SnowflakeClient(
            user=os.getenv('SF_USER'),
            pwd=os.getenv('SF_PWD'),
            account=os.getenv('SF_ACCOUNT'),
            role=os.getenv('SF_ROLE'),
            keep_alive=False
            )
    # upload data from H&M Dataset
    upload_data_to_snowflake(
        snowflake_client=sf_client,
        snowflake_db='EXPLORATION_DB', # change this with your own DB if you wish
        snowflake_schema='HM_RAW',
        snowflake_tables=['articles', 'customers', 'transactions_train'],
        data_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    )