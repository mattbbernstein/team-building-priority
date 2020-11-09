import os
import sqlite3 as sql
from dataingestion import *

DB_PATH = "data.db"

def main(db: sql.Connection):
    pass

if __name__ == "__main__":
    
    # TODO: Add option to force re-import

    # Connect to the database if it exists
    # otherwise import the data
    if os.path.exists(DB_PATH):
        db = connect_to_db(DB_PATH)
    else:
        db = import_data(DB_PATH)

    main(db)
    db.close()