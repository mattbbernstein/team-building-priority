import os
import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    # "OF" : "value_of",
    "DH" : "value_dh",
}

# TODO: Consider moving from sklearn to statsmodels

def main(db: sql.Connection):
    models = {}
    for position, table in TABLES.items():
        models[position] = linear_regression_fielders(db, table, position)
    models["Pitching"] = linear_regression_pitching(db)

    plot = False
    if plot:
        for position in models:
            plot_model(models[position], position)


    df = pd.read_sql_query("""
    SELECT 
        tr.Season, 
        tr.Team, 
        tr.W, 
        p.RAR AS p_rar,
        c.RAR AS c_rar,
        first.RAR AS first_rar,
        second.RAR AS second_rar,
        third.RAR AS third_rar,
        ss.RAR AS ss_rar,
        lf.RAR AS lf_rar,
        cf.RAR AS cf_rar,
        rf.RAR AS rf_rar,
        dh.RAR AS dh_rar
    FROM
        team_record AS tr
        INNER JOIN pitching AS p 
            ON 
                tr.Season = p.Season AND
                tr.Team = p.Team
        INNER JOIN value_c AS c 
            ON 
                tr.Season = c.Season AND
                tr.Team = c.Team
        INNER JOIN value_1b AS first 
            ON 
                tr.Season = first.Season AND
                tr.Team = first.Team
        INNER JOIN value_2b AS second 
            ON 
                tr.Season = second.Season AND
                tr.Team = second.Team
        INNER JOIN value_3b AS third 
            ON 
                tr.Season = third.Season AND
                tr.Team = third.Team
        INNER JOIN value_ss AS ss 
            ON 
                tr.Season = ss.Season AND
                tr.Team = ss.Team
        INNER JOIN value_lf AS lf 
            ON 
                tr.Season = lf.Season AND
                tr.Team = lf.Team
        INNER JOIN value_cf AS cf 
            ON 
                tr.Season = cf.Season AND
                tr.Team = cf.Team
        INNER JOIN value_rf AS rf 
            ON 
                tr.Season = rf.Season AND
                tr.Team = rf.Team
        INNER JOIN value_dh AS dh 
            ON 
                tr.Season = dh.Season AND
                tr.Team = dh.Team
    ORDER BY
        tr.Season DESC
    """, db)
    print(df)
    X = df[['p_rar','c_rar','first_rar','second_rar','third_rar','ss_rar','lf_rar','cf_rar','rf_rar','dh_rar']]
    X = sm.add_constant(X)
    Y = df['W']

    model = sm.OLS(Y, X).fit()
    print(model.summary())


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

    X = df[['Starting', 'Relieving']]
    X = sm.add_constant(X)
    Y = df['W']

    model = sm.OLS(Y, X).fit()
    print("Pitching:")
    print(model.summary())

    return model

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

    X = df[['Off', 'Def']]
    X = sm.add_constant(X)
    Y = df['W']

    model = sm.OLS(Y, X).fit()
    print("{}:".format(position))
    print(model.summary())

    return model

def plot_model(model: sm.OLS, title: str, fig: plt.Figure = None, save: bool = True) -> plt.Figure:

    # If an axes object was passed in don't save the figure
    # Otherwise, create an appropriate object
    if fig:
        save = False
    else:
        fig = plt.figure(figsize=(8,6))

    # model = model_data["model"]
    # raw = model_data["data"]
    # prediction = model.predict(raw)
    
    # axs.scatter(raw,win_data, color='blue', label="Raw Data")
    # axs.plot(raw,prediction, color='red', 
    #         label = r"Wins = {:0.3} * Runs + {:0.3}".format(model.coef_[0], model.intercept_) + "\n" + 
    #                 r"r$^2$ = {:0.3}".format(model.score(raw, win_data)))

    params = dict(model.params)
    data_cols = list(params.keys())[1:]
    fig = sm.graphics.plot_partregress_grid(model, exog_idx=data_cols, fig=fig)

    if save: 
        plt.savefig("Images/{}_plot.png".format(title))
        plt.close(fig)

    return fig

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