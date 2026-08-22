"""
Split out of EventSetBuilder.py (Phase 4 decomposition follow-up --
"break EventSetBuilder's 2184-line class into smaller, independently
testable units" per the refactor plan). ParamSet manages the
Monte-Carlo/midpoint/fmin parameter search that drives event-set
generation; it only ever needed `board`/`tracks`, never anything else
off `EventSetBuilder`, so it moves verbatim into its own module.
EventSetBuilder.py re-imports it (`from cribsandladders.param_set
import ParamSet`) so its own `self.paramSet = ParamSet(...)`
construction, and test_eventsetbuilder.py's `TestParamSet` (which
imports `ParamSet` directly), both keep working unchanged.

Persistence seam: every sqlite3 call here goes through
`self.board.config.optimizer_db_path` (a GameConfig property) instead
of a hardcoded 'etc/Optimizer.db' literal -- this is the fix
`test_monte_carlo_reads_from_configured_db_path_not_hardcoded_one`
pins down.
"""
import contextlib
import random as rd
import sqlite3 as sql
from datetime import datetime as dt
from io import StringIO

import pandas as pd


class ParamSet:
    """
    Manages parameters for event set generation and optimization.

    This class handles the configuration and optimization of various parameters
    that control how events are generated and placed on the game board.

    Attributes:
        board: Reference to the game board.
        tracks: List of tracks on the game board.
        params: List of parameter configurations.
    """

    def __init__(self, board, tracks):
        """
        Initialize the ParamSet with board and tracks.

        Args:
            board: The game board object.
            tracks: List of tracks on the game board.
        """
        # HOLE AND LENGTH BASED INTS:
        # baseopteventspertrack
        # ladderscanstartat
        # baseoptfirstchute
        # RELATIVE
        # candenergyskewdiminisher - Divides skew by this amount, maintain convergence
        # maxchuteoverdrivepct
        # holescompletetrackallowablecutoff
        # lengthhistogramscoringfactor
        # eventspacingdeviationfactor - Higher means more deviation in event spacing
        # eventspacinghistogramscoringfactor - Lower means less weight put upon it, MAX around 0.4 or 0.5!
        # candenergybufferdivider
        # move1allowanceratio INACTIVE
        # lengthhistogramscoringfactor - Lower means less weight put upon it
        # lengthovertimescoringfactor - Lower means less weight put upon it
        # disallowbelowsetlength - HELLA override! track spec only probs
        # maxorthoratio
        # minladdertochuteratio
        # minchutetoladderratio
        # twohitfreqimpedance - higher means more slowing of two-hit event combos
        self.board = board
        self.tracks = tracks
        self.params = []

    def intakeParams(self, instanceParams_df):
        """
        Load parameters from a DataFrame.

        Args:
            instanceParams_df: DataFrame containing parameter configurations.
        """
        self.params = []
        # Cursor thru params setting as needed
        for index, param_sr in instanceParams_df.iterrows():
            # Prioritize track override if exists
            self.params.append(dict(track_id=param_sr['track_id'], param=param_sr['param'], value=param_sr['value']))

    def intakeParamsFromDb(self, optimizerRunSet, optimizerRun):
        """
        Load parameters from the database.

        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                self.params = []
                paramsQuery_sb = StringIO()
                paramsQuery_sb.write("SELECT p.*")
                paramsQuery_sb.write(" from OptimizerRunTestParams p ")
                paramsQuery_sb.write("inner join OptimizerRuns o ")
                paramsQuery_sb.write("on o.OptimizerRun = p.OptimizerRun ")
                paramsQuery_sb.write("inner join OptimizerRunSets os ")
                paramsQuery_sb.write("on os.OptimizerRunSet = o.OptimizerRunSet ")
                paramsQuery_sb.write("where os.OptimizerRunSet = ? ")
                paramsQuery_sb.write("and o.OptimizerRun = ? ")
                params_df = pd.read_sql_query(paramsQuery_sb.getvalue(), sqlConn,
                                              params=[optimizerRunSet, optimizerRun])
                # Cursor thru params setting as needed
                for index, param_sr in params_df.iterrows():
                    # Prioritize track override if exists
                    self.params.append(dict(track_id=param_sr['Track_ID'], param=param_sr['Param'],
                                            value=param_sr['InstanceParamValue']))

    def midpointInitParams(self):
        """
        Initialize parameters with midpoint values for optimization.

        This sets up default parameter values that are in the middle of their
        allowed ranges to provide a good starting point for optimization.
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    # Retrieve base values from db
                    query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                    sqliteCursor.execute(query, [self.board.boardID, 0, 1])
                    boardparamranges_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                       columns=[d[0] for d in sqliteCursor.description])
                    if len(boardparamranges_df) == 0:
                        raise Exception("No param bounds found for board ID {}".format(self.board.boardID))

                    # Set params from ranges for board
                    self.params = []
                    # for index, param_sr in boardparamranges_df.iterrows():
                    #     # Prioritize track override if exists
                    #     if param_sr['isInt'] == 1:
                    #         curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                    #     else:
                    #         curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                    #     self.params.append(dict(track_id=0, param=param_sr['Param'], value=curVal))

                    for t in self.tracks:
                        # Try to retrieve overrides if exist
                        query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                        sqliteCursor.execute(query, [self.board.boardID, t.Track_ID, 1])
                        trackparamranges_df = pd.concat([pd.DataFrame(sqliteCursor.fetchall(),
                                                                      columns=[d[0] for d in sqliteCursor.description]),
                                                         boardparamranges_df])
                        trackparamranges_df.sort_values(['Param', 'Track_ID'], inplace=True,
                                                        ascending=False)

                        # Set params from ranges
                        prevParam = ""
                        for index, param_sr in trackparamranges_df.iterrows():
                            # Prioritize track override if exists
                            if param_sr['Param'] == prevParam: continue
                            if param_sr['isInt'] == 1:
                                curVal = (int(param_sr['LBound']) + int(param_sr['UBound'])) // 2
                            else:
                                curVal = (param_sr['LBound'] + param_sr['UBound']) / 2
                            self.params.append(dict(track_id=t.Track_ID, param=param_sr['Param'], value=curVal))
                            prevParam = param_sr['Param']

    def monteCarlo(self):
        """
        Perform a Monte Carlo simulation to optimize parameters.

        Randomly samples parameter values within their defined ranges to find
        optimal configurations for event generation.
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    # Retrieve base values from db
                    query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                    sqliteCursor.execute(query, [self.board.boardID, 0, 1])
                    boardparamranges_df = pd.DataFrame(sqliteCursor.fetchall(),
                                                       columns=[d[0] for d in sqliteCursor.description])
                    if len(boardparamranges_df) == 0:
                        raise Exception("No param bounds found for board ID {}".format(self.board.boardID))

                    # Set params from ranges for board
                    self.params = []
                    # for index, param_sr in boardparamranges_df.iterrows():
                    #     # Prioritize track override if exists
                    #     if param_sr['isInt'] == 1:
                    #         curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                    #     else:
                    #         curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                    #     self.params.append(dict(track_id=0, param=param_sr['Param'], value=curVal))

                    for t in self.tracks:
                        # Try to retrieve overrides if exist
                        query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID = ? AND Active = ?"
                        sqliteCursor.execute(query, [self.board.boardID, t.Track_ID, 1])
                        trackparamranges_df = pd.concat([pd.DataFrame(sqliteCursor.fetchall(),
                                                                      columns=[d[0] for d in sqliteCursor.description]),
                                                         boardparamranges_df])
                        trackparamranges_df.sort_values(['Param', 'Track_ID'], inplace=True,
                                                        ascending=False)

                        # Set params from ranges
                        prevParam = ""
                        for index, param_sr in trackparamranges_df.iterrows():
                            # Prioritize track override if exists
                            if param_sr['Param'] == prevParam: continue
                            if param_sr['isInt'] == 1:
                                curVal = rd.randint(int(param_sr['LBound']), int(param_sr['UBound']))
                            else:
                                curVal = rd.uniform(param_sr['LBound'], param_sr['UBound'])
                            self.params.append(dict(track_id=t.Track_ID, param=param_sr['Param'], value=curVal))
                            prevParam = param_sr['Param']

    def tempInsertParamsDb(self, optimizerRunSet, optimizerRun):
        """
        Temporarily store parameters in the database.

        Args:
            optimizerRunSet: Identifier for the optimizer run set.
            optimizerRun: Identifier for the specific optimizer run.
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    query = "INSERT INTO OptimizerRuns (OptimizerRunSet, OptimizerRun, Board_ID, Timestamp) VALUES (?,?,?,?)"
                    sqliteCursor.execute(query, [optimizerRunSet, optimizerRun, self.board.boardID,
                                                 dt.now().strftime('%m/%d/%y %H:%M:%S')])
                    sqlConn.commit()

                    params_df = pd.DataFrame.from_records(self.params)
                    # params_df.rename(columns={"track_id": "Track_ID", "param": "Param", "value": "InstanceParamValue"})
                    params_df['Board_ID'] = self.board.boardID
                    params_df['OptimizerRunSet'] = optimizerRunSet
                    params_df['OptimizerRun'] = optimizerRun

                    # Insert into data table
                    params_l = params_df.values.tolist()
                    paramsQuery_sb = StringIO()
                    paramsQuery_sb.write(
                        "INSERT INTO OptimizerRunTestParams(Track_ID, Param, InstanceParamValue, Board_ID, OptimizerRunSet, OptimizerRun) values(?,?,?,?,?,?)")
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    for index, record in params_df.iterrows():
                        sqliteCursor.execute(paramsQuery_sb.getvalue(), [record['track_id'], record['param'],
                                                                         record['value'], record['Board_ID'],
                                                                         record['OptimizerRunSet'],
                                                                         record['OptimizerRun']])
                    sqliteCursor.execute("END TRANSACTION")

                    sqlConn.commit()
                    paramsQuery_sb.close()

    def tempWriteMetricsToDb(self, evaluator):
        """
        Writes optimization metrics to the database for analysis and tracking.

        This method records various performance metrics from the evaluator to the optimizer database,
        which can be used for analyzing optimization performance and debugging.

        Args:
            evaluator: The evaluator object containing metrics to be recorded.

        Note:
            - Connects to the configured optimizer db (`board.config.optimizer_db_path`)
            - Creates necessary tables if they don't exist
            - Records metrics in a transaction for data consistency
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:

                    # Input metric info
                    metrics_df = pd.DataFrame.from_records(evaluator.results)
                    metrics_df['WeightedValue'] = metrics_df['Weighting'] * metrics_df['ResultValue']
                    metrics_df.drop(['Weighting', 'ResultValueIterative'], axis=1, inplace=True)
                    metrics_df['OptimizerRunSet'] = evaluator.optimizerRunSet
                    metrics_df['OptimizerRun'] = evaluator.optimizerRun
                    metrics_df['Board_ID'] = evaluator.board.boardID

                    # Insert into data table
                    metrics_l = metrics_df.values.tolist()
                    metricsQuery_sb = StringIO()
                    metricsQuery_sb.write("INSERT INTO OptimizerRunResults (")
                    cols = metrics_df.columns.values.tolist()
                    for c in range(0, len(cols) - 1):
                        metricsQuery_sb.write(cols[c])
                        metricsQuery_sb.write(", ")
                    metricsQuery_sb.write(cols[len(cols) - 1])
                    metricsQuery_sb.write(") VALUES (")
                    metricsQuery_sb.write(", ".join(['?'] * len(cols)))
                    metricsQuery_sb.write(")")
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    # sqliteCursor.execute("select * from  Testtest")
                    for index, record in metrics_df.iterrows():
                        sqliteCursor.execute(metricsQuery_sb.getvalue(), [record['Result'], record['ResultFlavour'],
                                                                          record['ResultValue'],
                                                                          record['WeightedValue'],
                                                                          record['OptimizerRunSet'],
                                                                          record['OptimizerRun'], record['Board_ID']])

                    sqliteCursor.execute("END TRANSACTION")
                    # sqliteCursor.executemany(metricsQuery_sb.getvalue(), metrics_df)
                    sqlConn.commit()
                    metricsQuery_sb.close()

    def tempWriteEvents(self, stats, optimizerRunSet, optimizerRun):
        """
        Writes event data to the database for a specific optimization run.

        This method records the current state of events (chutes and ladders) to the database
        for a given optimization run, allowing for tracking of how events evolve during optimization.

        Args:
            stats: Dictionary containing statistics about the current optimization state.
            optimizerRunSet: Identifier for the set of optimization runs.
            optimizerRun: Specific run identifier within the set.

        Note:
            - Records events in a transaction for data consistency
            - Tracks both chutes and ladders with their properties
            - Updates the database with the current best event configuration
        """
        with contextlib.closing(sql.connect(self.board.config.optimizer_db_path)) as sqlConn:
            with sqlConn:
                with contextlib.closing(sqlConn.cursor()) as sqliteCursor:
                    # Cache events hit in db
                    eventshit_df = pd.DataFrame.from_records(
                        [m.to_dict() for m in stats.moves if m.ladderorchuteamt != 0])
                    query = "INSERT INTO EventHit Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    sqliteCursor.execute("BEGIN TRANSACTION")
                    for idx, move in eventshit_df.iterrows():
                        sqliteCursor.execute(query, (optimizerRunSet, optimizerRun, self.board.boardID,
                                                     move['trial'], move['track_id'], move['threadnum'],
                                                     move['movenum'],
                                                     move['currpos'] - move['score'] + move['basescore'],
                                                     move['currpos'],
                                                     move['score'] - move['basescore']))

                    sqliteCursor.execute("END TRANSACTION")
                    sqlConn.commit()

    def modParamsForFmin(self, paramsSubset, fminParamsList):
        """
        Modifies parameters for function minimization during optimization.

        This method prepares and updates parameter values for use in the optimization
        process, converting between the optimization algorithm's format and the
        internal parameter representation.

        Args:
            paramsSubset: List of parameter names to be modified.
            fminParamsList: List of parameter values from the optimization algorithm.

        Returns:
            DataFrame: Updated parameters with new values for the specified subset.

        Note:
            - Handles parameter scaling and transformation if needed
            - Maintains parameter bounds and constraints
            - Updates the internal parameter state
        """
        allParams_df = pd.DataFrame.from_records(self.params)
        allParams_df.set_index(['param'], inplace=True)
        allParams_df.sort_index(inplace=True)
        for idx in range(0, len(paramsSubset)):
            param_df = allParams_df.loc[fminParamsList[idx]['param']]
            if isinstance(param_df, pd.Series): param_df = pd.DataFrame(param_df)
            for idx2, param_sr in param_df.iterrows():
                mask = (allParams_df.index == idx2) & (allParams_df['track_id'] == param_sr['track_id'])
                allParams_df.loc[mask, 'value'] = paramsSubset[idx]

        # Cursor thru params setting as needed
        self.params = []
        for index, param_sr in allParams_df.iterrows():
            # Prioritize track override if exists
            self.params.append(
                dict(track_id=param_sr['track_id'], param=index, value=param_sr['value']))

    def tryGetParam(self, track_ID, paramName, optional=False):
        """
        Retrieve a parameter value by track ID and parameter name.

        Args:
            track_ID: ID of the track to get the parameter for.
            paramName: Name of the parameter to retrieve.
            optional: If True, returns None if parameter not found instead of raising an error.

        Returns:
            The parameter value if found, None if optional and not found.

        Raises:
            Exception: If parameter is not found and optional is False.
        """
        record = next((param for param in self.params if param['param'] == paramName and
                       param['track_id'] == track_ID), None)
        if record is None:
            if optional: return 0
            raise Exception("{} not found for track_ID {}".format(paramName, track_ID))

        return record['value']

    def tryModParam(self, track_ID, paramName, newValue):
        """
        Modify a parameter value.

        Args:
            track_ID: ID of the track containing the parameter.
            paramName: Name of the parameter to modify.
            newValue: New value to set for the parameter.

        Returns:
            bool: True if parameter was found and modified, False otherwise.
        """
        # Iterate through the list of dictionaries
        for record in self.params:
            if record['track_id'] == track_ID and record['param'] == paramName:
                record['value'] = newValue
                break  # Exit the loop once the record is found and modified
