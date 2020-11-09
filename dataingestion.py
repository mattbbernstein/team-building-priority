""" dataingestion.py

Contains the functions to ingest data from the datasets
and build the database
"""

import pandas as pd
import sqlite3 as sql
import os, glob

def connect_to_db(db_file: str) -> sql.Connection:
    """Connects to the given database"""

    conn = None
    try:
        conn = sql.connect(db_file)
    except Exception as e:
        print(e)
    return conn

def import_csv(csv_file: str, db: sql.Connection):
    """ Reads in a csv and creates a table in db
        with the same name as the CSV file
    """


    df = pd.read_csv(csv_file)
    basename = os.path.basename(csv_file)
    table_name = os.path.splitext(basename)[0]
    df.to_sql(table_name, db, if_exists="append", index=True)

def import_data(db_file: str, dataset_dir: str = "datasets") -> sql.Connection:
    """ Imports all the files in dataset_dir into a new database.
        
        Arguments:
            db_dile (str): String path to database
            dataset_dir (str): String path to folder containing datasets (Default: ./datasets)

        Returns:
            db (sqlite3.Connection): Database connection (None if error)
        
        Note: Erases and removes any database file that already exists
    """


    db = connect_to_db(db_file)
    if db:

        datasets = glob.glob(os.path.join(dataset_dir, "*.csv"))
        for dataset in datasets:
            import_csv(dataset, db)
    
    return db