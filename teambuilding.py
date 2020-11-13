import os
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
from dataingestion import *
from sklearn.linear_model import LinearRegression

DB_PATH = "data.db"
TABLES = {
    "C"  : "value_c",
    "1B" : "value_1b",
    "2B" : "value_2b",
    "3B" : "value_3b",
    "SS" : "value_ss",
    "LF" : "value_lf",
    "CF" : "value_cf",
    "RF" : "value_rf",
    "OF" : "value_of",
    "DH" : "value_dh",
}


def main(db: sql.Connection):
    (m1, m2) = linear_regression_fielders(db, TABLES["C"], "C")

def linear_regression_fielders(db: sql.Connection, table: str, position: str):
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
    or_sq = off_model.score(r_off,wins)
    ointercept = off_model.intercept_
    oslope = off_model.coef_[0]

    print("Offense:\nr^2 = {:0.3}\nWins = {:0.3} * Off + {:0.3}\n".format(or_sq, oslope, ointercept))
    owins_pred = off_model.predict(r_off)

    def_model = LinearRegression().fit(r_def, wins)
    dr_sq = def_model.score(r_def,wins)
    dintercept = def_model.intercept_
    dslope = def_model.coef_[0]

    print("Defense\nr^2 = {:0.3}\nWins = {:0.3} * Off + {:0.3}\n".format(dr_sq, dslope, dintercept))
    dwins_pred = def_model.predict(r_def)

    # Plotting the data
    plt.rcParams["figure.figsize"] = (10,8)
    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=0.4)

    axs[0].scatter(r_off,wins, color='blue', label="Raw Data")
    axs[0].plot(r_off,owins_pred, color='red', label="Fit Line\nW = {:0.3}*Off + {:0.3}".format(oslope, ointercept))
    axs[0].set_title("Offensive Runs")
    axs[0].legend(loc="lower right")

    axs[1].scatter(r_def,wins, color='blue', label="Raw Data")
    axs[1].plot(r_def,dwins_pred, color='red', label="Fit Line\nW = {:0.3}*Off + {:0.3}".format(dslope, dintercept))
    axs[1].set_title("Defensive Runs")
    axs[1].legend(loc="lower right")

    # Formatting the axis
    for ax in axs.flat:
        ax.set(xlabel='Runs', ylabel='Team Wins')
        ax.label_outer()
    plt.suptitle("{}: Runs Created vs Team Wins".format(position))
    # plt.legend(labels=["Offensive Runs", "Defensive Runs","Offensive Prediction", "Defensive Prediction"])

    plt.savefig("{}_plot.png".format(position))

    return (off_model, def_model)

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