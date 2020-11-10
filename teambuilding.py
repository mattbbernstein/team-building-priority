import os
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
from dataingestion import *
from sklearn.linear_model import LinearRegression

DB_PATH = "data.db"

def main(db: sql.Connection):
    pass

def linear_regression_fielders(db: sql.Connection, table: str):
    """ Runs a simple linear regression for Non-pitchers

        Runs a linear regresssion for Offensive and Defensive RC vs
        team wins. Saves a scatter plot with both sets of data and
        regression lines. Returns the two models

        Arguments:
            db (sqlite.Connection): Database connection
            table (str): Name of table to run regression on

        Returns:
            (model_off, model_def): Offensive and defensive models
    """
    table="value_1b"
    df = pd.read_sql_query("""
    SELECT 
        tr.Season, 
        tr.Team, 
        tr.W, 
        v.Off, 
        v.Def
    FROM
        team_record AS tr,
        {} AS v
    WHERE
        tr.Season = v.Season
        AND
        tr.Team = v.Team
    ORDER BY
        tr.Season DESC
    """.format(table),
    db)
    wins = df["W"].values
    r_off = df["Off"].values.reshape(-1,1)
    r_def = df["Def"].values.reshape(-1,1)
    
    off_model = LinearRegression().fit(r_off, wins)
    r_sq = off_model.score(r_off,wins)
    intercept = off_model.intercept_
    slope = off_model.coef_[0]

    print("Offense:\nr^2 = {:0.3}\nWins = {:0.3} * Off + {:0.3}\n".format(r_sq, slope, intercept))
    owins_pred = off_model.predict(r_off)

    def_model = LinearRegression().fit(r_def, wins)
    r_sq = def_model.score(r_def,wins)
    intercept = def_model.intercept_
    slope = def_model.coef_[0]

    print("Defense\nr^2 = {:0.3}\nWins = {:0.3} * Off + {:0.3}\n".format(r_sq, slope, intercept))
    dwins_pred = def_model.predict(r_def)

    plt.scatter(r_off,wins, color='blue')
    plt.scatter(r_def,wins, color='red')
    plt.plot(r_off,owins_pred, color='black')
    plt.plot(r_def,dwins_pred, color='green')
    plt.savefig("1b_plot.png")

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