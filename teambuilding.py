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

# TODO: Consider moving from sklearn to statsmodels

def main(db: sql.Connection):
    models = {}
    for position, table in TABLES.items():
        models[position] = linear_regression_fielders(db, table, position)
    models["Pitching"] = linear_regression_pitching(db)

    plot_model(models["Pitching"]["Starting"], models["Pitching"]["W"], "Pitching")

def linear_regression_pitching(db:sql.Connection) -> LinearRegression:
    """ Runs a simple linear regression for Pitchers

        Runs a linear regresssion for Pitching RC vs
        team wins. Saves a scatter plot with both sets of data and
        regression lines. Returns the two models.
        This checks starting pitching, relief pitching, and
        then total pitching

        Arguments:
            db (sqlite.Connection): Database connection

        Returns:
            Model dictionary: Pitching models
    """

    df = pd.read_sql_query("""
    SELECT 
        tr.Season, 
        tr.Team, 
        tr.W, 
        p.Starting, 
        p.Relieving,
        p.RAR
    FROM
        team_record AS tr,
        pitching AS p
    WHERE
        tr.Season = p.Season
        AND
        tr.Team = p.Team
    ORDER BY
        tr.Season DESC
    """,
    db)
    wins = df["W"].values
    r_start = df["Starting"].values.reshape(-1,1)
    r_relief = df["Relieving"].values.reshape(-1,1)
    r_tot = df["RAR"].values.reshape(-1,1)

    start_model = LinearRegression().fit(r_start, wins)
    sr_sq = start_model.score(r_start,wins)
    sintercept = start_model.intercept_
    sslope = start_model.coef_[0]

    print("Analysis: Pitching")

    print("Starting: r^2 = {:0.3} -- Wins = {:0.3} * Starting + {:0.3}".format(sr_sq, sslope, sintercept))

    relief_model = LinearRegression().fit(r_relief, wins)
    rr_sq = relief_model.score(r_relief,wins)
    rintercept = relief_model.intercept_
    rslope = relief_model.coef_[0]

    print("Relief:   r^2 = {:0.3} -- Wins = {:0.3} * Relief + {:0.3}".format(rr_sq, rslope, rintercept))

    tot_model = LinearRegression().fit(r_tot, wins)
    tr_sq = tot_model.score(r_tot,wins)
    tintercept = tot_model.intercept_
    tslope = tot_model.coef_[0]

    print("Total:    r^2 = {:0.3} -- Wins = {:0.3} * RAR + {:0.3}".format(tr_sq, tslope, tintercept))

    print("--------------------------------------------------\n")

    ret_dict = {
        "Starting": {"model": start_model, "data": r_start},
        "Relief":   {"model": relief_model, "data": r_relief},
        "Total":    {"model": tot_model, "data": r_tot},
        "W": wins
    }
    return ret_dict

def linear_regression_fielders(db: sql.Connection, table: str, position: str) -> (LinearRegression, LinearRegression):
    """ Runs a simple linear regression for Non-pitchers

        Runs a linear regresssion for Offensive and Defensive RC vs
        team wins. Saves a scatter plot with both sets of data and
        regression lines. Returns the two models

        Arguments:
            db (sqlite.Connection): Database connection
            table (str): Name of table to run regression on

        Returns:
            (model_off, model_def, model_total): Offensive, defensive, and overall models
    """

    df = pd.read_sql_query("""
    SELECT 
        tr.Season, 
        tr.Team, 
        tr.W, 
        v.Off, 
        v.Def,
        v.RAR
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
    r_tot = df["RAR"].values.reshape(-1,1)
    
    off_model = LinearRegression().fit(r_off, wins)
    or_sq = off_model.score(r_off,wins)
    ointercept = off_model.intercept_
    oslope = off_model.coef_[0]

    print("Analysis: {}".format(position))

    print("Offense: r^2 = {:0.3} -- Wins = {:0.3} * Off + {:0.3}".format(or_sq, oslope, ointercept))

    def_model = LinearRegression().fit(r_def, wins)
    dr_sq = def_model.score(r_def,wins)
    dintercept = def_model.intercept_
    dslope = def_model.coef_[0]

    print("Defense: r^2 = {:0.3} -- Wins = {:0.3} * Off + {:0.3}".format(dr_sq, dslope, dintercept))

    tot_model = LinearRegression().fit(r_tot, wins)
    tr_sq = tot_model.score(r_tot,wins)
    tintercept = tot_model.intercept_
    tslope = tot_model.coef_[0]

    print("Total:   r^2 = {:0.3} -- Wins = {:0.3} * RAR + {:0.3}".format(tr_sq, tslope, tintercept))

    print("--------------------------------------------------\n")

    ret_dict = {
        "Offense":  {"model": off_model, "data": r_off},
        "Defense":  {"model": def_model, "data": r_def},
        "Total":    {"model": tot_model, "data": r_tot},
        "W": wins
    }
    return ret_dict

def plot_model(model_data: dict, win_data: np.ndarray, title: str, axs: plt.Axes = None, save: bool = True) -> plt.Axes:

    # If an axes object was passed in don't save the figure
    # Otherwise, create an appropriate object
    if axs:
        save = False
    else:
        _, axs = plt.subplots(1)

    model = model_data["model"]
    raw = model_data["data"]
    prediction = model.predict(raw)
    
    axs.scatter(raw,win_data, color='blue', label="Raw Data")
    axs.plot(raw,prediction, color='red', 
            label = r"Wins = {:0.3} * Runs + {:0.3}".format(model.coef_[0], model.intercept_) + "\n" + 
                    r"r$^2$ = {:0.3}".format(model.score(raw, win_data)))
    
    axs.set_title("Runs")
    axs.legend(loc="lower right")
    axs.set(xlabel='Runs', ylabel='Team Wins')

    if save: plt.savefig("Images/{}_plot.png".format(title))

    return axs

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