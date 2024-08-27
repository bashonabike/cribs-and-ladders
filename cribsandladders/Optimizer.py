import pandas as pd
import numpy as np
import sqlite3 as sql
from io import StringIO
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import random as rd
import os
import datetime as dt
from sklearn.metrics import mean_squared_error
import mystic
from mystic.solvers import fmin
from mystic.monitors import VerboseMonitor

import game_params as gp



class Optimizer:
    def __init__(self, board, optimizerRunSet):
        self.prevParams = []
        self.params = []
        self.prevResults = []
        self.freshResults = []
        self.bestPostIterParams = []
        self.bestPostFminParams = []
        self.board = board
        self.optimizerRunSet = optimizerRunSet
        self.sqlConn = sql.connect('etc/Optimizer.db')
        self.sqliteCursor = self.sqlConn.cursor()
        self.trainFullSet_df = None
        self.paramsTrainSet_df = None
        self.resultsTrainSet_df = None
        self.pairings_df, self.absoluteBounds = None, None
        self.retrievePairingsSettings()
        self.fminParamsList = []

    def retrieveTrainData(self):
        #Retrieve eligible param settings from db
        boardID = self.board.boardID
        query = "SELECT Param, MIN(Track_ID) AS Elig_Track_ID FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID >= ? GROUP BY Param"
        trackparamranges_df = pd.read_sql_query(query, self.sqlConn, params=[self.board.boardID, 0])
        # self.sqliteCursor.execute(query, [self.board.boardID, 0])
        # trackparamranges_df = pd.DataFrame(self.sqliteCursor.fetchall(),
        #                                               columns=[d[0] for d in self.sqliteCursor.description])
        trackparamranges_df.sort_values(['Param'], inplace=True)

        #Build query to retrieve train/test set params
        paramsQuery_sb = StringIO()
        paramsQuery_sb.write("select os.OptimizerRunSet, o.OptimizerRun, p.Board_ID")
        for t in self.board.tracks:
            for index, param_sr in trackparamranges_df.iterrows():
                if param_sr['Elig_Track_ID'] in (0, t.Track_ID):
                    paramsQuery_sb.write(" , IFNULL(sum(CASE WHEN Track_ID = ")
                    paramsQuery_sb.write(str(t.Track_ID))
                    paramsQuery_sb.write(" AND Param = '")
                    paramsQuery_sb.write(param_sr['Param'])
                    paramsQuery_sb.write("' THEN p.InstanceParamValue END), 0) AS ")
                    paramsQuery_sb.write(param_sr['Param'])
                    paramsQuery_sb.write("_T")
                    paramsQuery_sb.write(str(t.Track_ID))

        paramsQuery_sb.write(" from OptimizerRunTestParams p ")
        paramsQuery_sb.write("inner join OptimizerRuns o ")
        paramsQuery_sb.write("on o.OptimizerRun = p.OptimizerRun ")
        paramsQuery_sb.write("inner join OptimizerRunSets os ")
        paramsQuery_sb.write("on os.OptimizerRunSet = o.OptimizerRunSet ")
        paramsQuery_sb.write("where os.OptimizerRunSet = ? ")
        paramsQuery_sb.write("group by  os.OptimizerRunSet, o.OptimizerRun, p.Board_ID ")
        self.paramsTrainSet_df = pd.read_sql_query(paramsQuery_sb.getvalue(), self.sqlConn, params=[self.optimizerRunSet])
        # self.sqliteCursor.execute(paramsQuery_sb.getvalue(), [self.optimizerRunSet])
        # paramsTrainSet_df = pd.DataFrame(self.sqliteCursor.fetchall(),
        #                                    columns=[d[0] for d in self.sqliteCursor.description])
        self.paramsTrainSet_df.sort_values(['OptimizerRun'], inplace=True)
        paramsQuery_sb.close()

        #Retrieve optimizer result train set data
        query = ("SELECT r.Result FROM OptimizerRunResults r INNER JOIN OptimizerRuns op " +
                 " ON op.OptimizerRun = r.OptimizerRun INNER JOIN OptimizerRunSets os on " +
                " os.OptimizerRunSet = op.OptimizerRunSet WHERE os.OptimizerRunSet = ? GROUP BY r.Result")
        posResults_df = pd.read_sql_query(query, self.sqlConn, params=[self.optimizerRunSet])
        posResults_df.sort_values(['Result'], inplace=True)
        resultsQuery_sb = StringIO()
        resultsQuery_sb.write("select os.OptimizerRunSet, o.OptimizerRun, r.Board_ID")
        for index, result_sr in posResults_df.iterrows():
            resultsQuery_sb.write(" , IFNULL(sum(CASE WHEN Result = '")
            resultsQuery_sb.write(result_sr['Result'])
            resultsQuery_sb.write("' THEN r.ResultValue END), 0) AS ")
            resultsQuery_sb.write(result_sr['Result'])

        resultsQuery_sb.write(" from OptimizerRunResults r ")
        resultsQuery_sb.write("inner join OptimizerRuns o ")
        resultsQuery_sb.write("on o.OptimizerRun = r.OptimizerRun ")
        resultsQuery_sb.write("inner join OptimizerRunSets os ")
        resultsQuery_sb.write("on os.OptimizerRunSet = o.OptimizerRunSet ")
        resultsQuery_sb.write("where os.OptimizerRunSet = ? ")
        resultsQuery_sb.write("group by  os.OptimizerRunSet, o.OptimizerRun, r.Board_ID ")
        self.resultsTrainSet_df = pd.read_sql_query(resultsQuery_sb.getvalue(), self.sqlConn, params=[self.optimizerRunSet])
        # self.sqliteCursor.execute(resultsQuery_sb.getvalue(), [self.optimizerRunSet])
        # resultsTrainSet_df = pd.DataFrame(self.sqliteCursor.fetchall(),
        #                                    columns=[d[0] for d in self.sqliteCursor.description])
        self.resultsTrainSet_df.sort_values(['OptimizerRun'], inplace=True)
        temp = resultsQuery_sb.getvalue()
        resultsQuery_sb.close()

        #Join together into single set
        self.trainFullSet_df = pd.merge(
                        self.paramsTrainSet_df,      # First DataFrame
                        self.resultsTrainSet_df,     # Second DataFrame
                        on=['OptimizerRunSet', 'OptimizerRun', 'Board_ID'],  # Columns to join on
                        how='inner'  # Type of join: 'inner', 'left', 'right', or 'outer'
                        )

    def retrievePairingsSettings(self):
        query = "SELECT * FROM OptimizerParamPairings"
        self.pairings_df = pd.read_sql_query(query, self.sqlConn)
        self.pairings_df.set_index(['Result'], inplace=True)
        self.pairings_df.sort_index(inplace=True)

        query = "SELECT * FROM BoardTrackHints WHERE Board_ID = ? AND Track_ID IN(?,?)"
        self.absoluteBounds = pd.read_sql_query(query, self.sqlConn, params=[self.board.boardID, -1, -2])
        self.absoluteBounds.set_index(['Param'], inplace=True)
        self.absoluteBounds.sort_index(inplace=True)


    def runGBM(self):
        #Run GBM train
        #NOTE: we are using results as X since we want to train regression model to output optimal input params to Cribs model
        X = self.resultsTrainSet_df.drop(['OptimizerRunSet', 'OptimizerRun' ,'Board_ID'], axis=1)
        # X = self.resultsTrainSet_df.drop(['OptimizerRunSet', 'OptimizerRun' ,'Board_ID', 'eventsHitLengthDistribution_curvefit', 'eventSpacingHist_curvefit'], axis=1)
        y = self.paramsTrainSet_df.drop(['OptimizerRunSet', 'OptimizerRun' ,'Board_ID'], axis=1)
        testtotraindataratio = self.setParamFromBounds(gp.testtotraindataratio_bnds)
        trainrandomstate = self.setParamFromBounds(gp.trainrandomstate_bnds)
        trainlearningrate = self.setParamFromBounds(gp.trainlearningrate_bnds)
        trainnumestimators = self.setParamFromBounds(gp.trainnumestimators_bnds)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=testtotraindataratio,
                                                            random_state=trainrandomstate)

        lgb_model = lgb.LGBMRegressor(learning_rate=trainlearningrate, n_estimators=trainnumestimators)
        model = MultiOutputRegressor(lgb_model)
        model.fit(X_train, y_train)

        #Compare predictions w/ actuals
        y_pred = pd.DataFrame(model.predict(X_test))

        # Sum of squared differences
        sum_squared_diff = mean_squared_error(y_test, y_pred)

        # y_pred.to_csv("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\y_predicts\\y_pred-{}-{}-{}-{}-{}.csv".format(
        #                                              dt.datetime.now().strftime("%Y-%m-%d--%H-%M"), testtotraindataratio, trainrandomstate, trainlearningrate,
        #                                              trainnumestimators))
        # y_test.to_csv("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\y_predicts\\y_test-{}-{}-{}-{}-{}.csv".format(
        #                                              dt.datetime.now().strftime("%Y-%m-%d--%H-%M"), testtotraindataratio, trainrandomstate, trainlearningrate,
        #                                              trainnumestimators,))

        print(str(sum_squared_diff))

        test = pd.DataFrame(columns=X_test.columns)
        test.loc[len(test)] = [0.0] * test.shape[1]

        opt = pd.DataFrame(model.predict(test), columns=y_test.columns)

        # opt.to_csv("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\y_predicts\\opt-{}-{}-{}-{}-{}.csv".format(
        #                                              dt.datetime.now().strftime("%Y-%m-%d--%H-%M"), testtotraindataratio, trainrandomstate, trainlearningrate,
        #                                              trainnumestimators,))

        print(opt.to_dict())

        # optFormatted = pd.DataFrame(columns=['track_id', 'Param', 'value'])
        optFormatted = opt.transpose()
        optFormatted['track_id'] = optFormatted.index.str[-2:].astype(int)
        optFormatted['Param'] = optFormatted.index.str[:-4]
        optFormatted['value'] = optFormatted[0]

        print(optFormatted.to_dict())


        return optFormatted, model, sum_squared_diff

    def testGBMOnPairings(self, optPairings, param):
        #TEMP!!
        resultsPreChanges = self.resultsTrainSet_df[self.resultsTrainSet_df['OptimizerRun'] < 1725]
        paramsPreChanges = self.paramsTrainSet_df[self.paramsTrainSet_df['OptimizerRun'] < 1725]
        X = resultsPreChanges[optPairings]
        y = paramsPreChanges.filter(like=param, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=40)

        lgb_model = lgb.LGBMRegressor(learning_rate=.001, n_estimators=400)
        model = MultiOutputRegressor(lgb_model)
        model.fit(X_train, y_train)

        # Compare predictions w/ actuals
        y_pred = pd.DataFrame(model.predict(X_test))

        # Sum of squared differences
        sum_squared_diff = mean_squared_error(y_test, y_pred)

        # y_pred.to_csv("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\y_predicts\\y_pred-{}-{}-{}-{}-{}.csv".format(
        #                                              dt.datetime.now().strftime("%Y-%m-%d--%H-%M"), testtotraindataratio, trainrandomstate, trainlearningrate,
        #                                              trainnumestimators))
        # y_test.to_csv("C:\\Users\\Dell 5290\\Documents\\cribs-and-ladders\\etc\\y_predicts\\y_test-{}-{}-{}-{}-{}.csv".format(
        #                                              dt.datetime.now().strftime("%Y-%m-%d--%H-%M"), testtotraindataratio, trainrandomstate, trainlearningrate,
        #                                              trainnumestimators,))

        print(str(sum_squared_diff))
        return sum_squared_diff

    def runIncrIteration (self, freshParams, freshResults):
        self.params = freshParams
        self.freshResults = freshResults

        params_df = pd.DataFrame.from_records(freshParams)
        params_df.set_index(['param'], inplace=True)
        params_df.sort_index(inplace=True)

        #Determine which result to target
        results_df = pd.DataFrame.from_records(freshResults)
        results_df['EffIterResult'] = np.where(results_df['ResultValueIterative'].notna(),
                                               results_df['ResultValueIterative'], results_df['ResultValue'])
        #NOTE: we use absolutified for WeighedResult since we are trying to find greatest offender
        results_df['WeighedResult'] = results_df['ResultValue']*results_df['Weighting']
        results_df.sort_values(['WeighedResult'], ascending=False)
        target, targetPairings_l = None, []
        trackwise, track_ID, resultType = False, -1, ""

        for r_idx, result_sr in results_df.iterrows():
            paramMaxed = False
            if "_T" in result_sr['Result']:
                trackwise = True
                start_index = result_sr['Result'].find("_T") + len("_T")
                track_ID = result_sr['Result'][start_index:]
                resultType = result_sr['Result'][:start_index - len("_T")]
            else:
                resultType = result_sr['Result']
            pairings = self.pairings_df.loc[resultType]
            if type(pairings) == pd.DataFrame:
                for pair_sr in pairings:
                    target = result_sr
                    targetPairings_l.append(pair_sr.to_dict())
            else:
                target = result_sr
                targetPairings_l.append(pairings.to_dict())
            if target is None: continue

            #Incr or decr paired params dep on whether inverse or not
            reverse = -1 if target['EffIterResult'] < 0 else 1
            for pairing_dct in targetPairings_l:
                inverse = -1 if pairing_dct['Inverse'] > 0 else 1
                absbounds_sr = self.absoluteBounds.loc[pairing_dct['Param']]
                targetParams_df = params_df.loc[pairing_dct['Param']]
                for idx, targetParam_sr in targetParams_df.iterrows():
                    if trackwise and targetParam_sr['track_id'] != track_ID: continue
                    newVal = targetParam_sr['value']*(1.0 - reverse*inverse*gp.changepctperiteration)
                    if newVal < absbounds_sr['LBound'] or newVal > absbounds_sr['UBound']:
                        paramMaxed = True
                        break
                    targetParam_sr['InstanceParamValue'] = newVal
                if paramMaxed: break
            if paramMaxed: continue
            else: break

        #Close up
        params_df['param'] = params_df.index
        self.params = []
        for idx, sr in params_df.iterrows():
            paramLine = dict(track_id=sr['track_id'], param=sr['param'], value=sr['value'])
            self.params.append(paramLine)
        self.prevParams = self.params
        self.prevResults = self.freshResults
        return self.params

    def detWeighedScoring(self, freshResults):
        results_df = pd.DataFrame.from_records(freshResults)
        results_df['WeighedResult'] = results_df['ResultValue']*results_df['Weighting']
        return results_df['WeighedResult'].sum()

    def getFminStarterParams(self):
        starterParams_l = []
        bestParams_df = pd.DataFrame.from_records(self.bestPostIterParams)
        bestParams_df.set_index(['Param'], inplace=True)
        for paramName in self.fminParamsList:
            starterParams_l.append(bestParams_df.loc(paramName)['InstanceParamValue'])

        return starterParams_l

    def getFminBounds(self):
        selectBounds_l = []
        for paramName in self.fminParamsList:
            bounds_sr = self.absoluteBounds.loc(paramName)
            selectBounds_l.append((bounds_sr['LBound'], bounds_sr['UBound']))

        return selectBounds_l

    def setBestFminParams(self, bestFminParams):
        bestIter_df = pd.DataFrame.from_records( self.bestPostIterParams)
        bestIter_df.set_index(['Param'], inplace=True)
        for idx in range(len(self.fminParamsList)):
            param_sr = bestIter_df.loc(self.fminParamsList[idx])
            param_sr['InstanceParamValue'] = bestFminParams[idx]

        self.bestPostFminParams = []
        for idx, sr in bestIter_df.iterrows():
            paramLine = dict(track_id=sr['track_id'], param=sr['param'], value=sr['value'])
            self.bestPostFminParams.append(paramLine)



    def setBestIterParams(self, bestParams):
        self.bestPostIterParams = bestParams


    def setupFminParamsList(self, sampleParams):
        allPairings_sr = self.pairings_df.loc[self.pairings_df['Result'] == "ALL"]
        sampleParams_df = pd.DataFrame.from_records(sampleParams)
        sampleParams_df.set_index(['Param'], inplace=True)
        self.fminParamsList = []
        for pairing_sr in allPairings_sr:
            for param_sr in sampleParams_df[sampleParams_df.index.str.startswith(pairing_sr['Param'])]:
                self.fminParamsList.append(param_sr.index)

    def setParamFromBounds(self, gameParamBounds):
        if gameParamBounds[2]:
            return rd.randint(int(gameParamBounds[0]), int(gameParamBounds[1]))
        else:
            return rd.uniform(gameParamBounds[0], gameParamBounds[1])






