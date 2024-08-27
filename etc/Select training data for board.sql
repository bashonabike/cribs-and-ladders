

select os.OptimizerRunSet, p.OptimizerRun, p.Board_ID

 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'baseopteventspertrack' THEN p.InstanceParamValue END), 0) AS baseopteventspertrack_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'baseoptfirstchute' THEN p.InstanceParamValue END), 0) AS baseoptfirstchute_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'candenergybufferdivider' THEN p.InstanceParamValue END), 0) AS candenergybufferdivider_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'candenergyskewdiminisher' THEN p.InstanceParamValue END), 0) AS candenergyskewdiminisher_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'disallowbelowsetlength' THEN p.InstanceParamValue END), 0) AS disallowbelowsetlength_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'eventspacingdeviationfactor' THEN p.InstanceParamValue END), 0) AS eventspacingdeviationfactor_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'eventspacinghistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS eventspacinghistogramscoringfactor_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'holescompletetrackallowablecutoff' THEN p.InstanceParamValue END), 0) AS holescompletetrackallowablecutoff_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'ladderscanstartat' THEN p.InstanceParamValue END), 0) AS ladderscanstartat_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'lengthhistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS lengthhistogramscoringfactor_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'lengthovertimescoringfactor' THEN p.InstanceParamValue END), 0) AS lengthovertimescoringfactor_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'maxchuteoverdrivepct' THEN p.InstanceParamValue END), 0) AS maxchuteoverdrivepct_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'maxorthoratio' THEN p.InstanceParamValue END), 0) AS maxorthoratio_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'minchutetoladderratio' THEN p.InstanceParamValue END), 0) AS minchutetoladderratio_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'minladdertochuteratio' THEN p.InstanceParamValue END), 0) AS minladdertochuteratio_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'move1allowanceratio' THEN p.InstanceParamValue END), 0) AS move1allowanceratio_T14
 , IFNULL(sum(CASE WHEN Track_ID = 14 AND Param = 'twohitfreqimpedance' THEN p.InstanceParamValue END), 0) AS twohitfreqimpedance_T14
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'baseopteventspertrack' THEN p.InstanceParamValue END), 0) AS baseopteventspertrack_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'baseoptfirstchute' THEN p.InstanceParamValue END), 0) AS baseoptfirstchute_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'candenergybufferdivider' THEN p.InstanceParamValue END), 0) AS candenergybufferdivider_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'candenergyskewdiminisher' THEN p.InstanceParamValue END), 0) AS candenergyskewdiminisher_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'disallowbelowsetlength' THEN p.InstanceParamValue END), 0) AS disallowbelowsetlength_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'eventspacingdeviationfactor' THEN p.InstanceParamValue END), 0) AS eventspacingdeviationfactor_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'eventspacinghistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS eventspacinghistogramscoringfactor_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'holescompletetrackallowablecutoff' THEN p.InstanceParamValue END), 0) AS holescompletetrackallowablecutoff_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'ladderscanstartat' THEN p.InstanceParamValue END), 0) AS ladderscanstartat_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'lengthhistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS lengthhistogramscoringfactor_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'lengthovertimescoringfactor' THEN p.InstanceParamValue END), 0) AS lengthovertimescoringfactor_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'maxchuteoverdrivepct' THEN p.InstanceParamValue END), 0) AS maxchuteoverdrivepct_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'maxorthoratio' THEN p.InstanceParamValue END), 0) AS maxorthoratio_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'minchutetoladderratio' THEN p.InstanceParamValue END), 0) AS minchutetoladderratio_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'minladdertochuteratio' THEN p.InstanceParamValue END), 0) AS minladdertochuteratio_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'move1allowanceratio' THEN p.InstanceParamValue END), 0) AS move1allowanceratio_T15
 , IFNULL(sum(CASE WHEN Track_ID = 15 AND Param = 'twohitfreqimpedance' THEN p.InstanceParamValue END), 0) AS twohitfreqimpedance_T15
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'baseopteventspertrack' THEN p.InstanceParamValue END), 0) AS baseopteventspertrack_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'baseoptfirstchute' THEN p.InstanceParamValue END), 0) AS baseoptfirstchute_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'candenergybufferdivider' THEN p.InstanceParamValue END), 0) AS candenergybufferdivider_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'candenergyskewdiminisher' THEN p.InstanceParamValue END), 0) AS candenergyskewdiminisher_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'disallowbelowsetlength' THEN p.InstanceParamValue END), 0) AS disallowbelowsetlength_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'eventspacingdeviationfactor' THEN p.InstanceParamValue END), 0) AS eventspacingdeviationfactor_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'eventspacinghistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS eventspacinghistogramscoringfactor_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'holescompletetrackallowablecutoff' THEN p.InstanceParamValue END), 0) AS holescompletetrackallowablecutoff_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'ladderscanstartat' THEN p.InstanceParamValue END), 0) AS ladderscanstartat_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'lengthhistogramscoringfactor' THEN p.InstanceParamValue END), 0) AS lengthhistogramscoringfactor_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'lengthovertimescoringfactor' THEN p.InstanceParamValue END), 0) AS lengthovertimescoringfactor_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'maxchuteoverdrivepct' THEN p.InstanceParamValue END), 0) AS maxchuteoverdrivepct_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'maxorthoratio' THEN p.InstanceParamValue END), 0) AS maxorthoratio_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'minchutetoladderratio' THEN p.InstanceParamValue END), 0) AS minchutetoladderratio_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'minladdertochuteratio' THEN p.InstanceParamValue END), 0) AS minladdertochuteratio_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'move1allowanceratio' THEN p.InstanceParamValue END), 0) AS move1allowanceratio_T16
 , IFNULL(sum(CASE WHEN Track_ID = 16 AND Param = 'twohitfreqimpedance' THEN p.InstanceParamValue END), 0) AS twohitfreqimpedance_T16




from OptimizerRunTestParams p
inner join OptimizerRuns o
on o.OptimizerRun = p.OptimizerRun
inner join OptimizerRunSets os
on os.OptimizerRunSet = o.OptimizerRunSet
where os.OptimizerRunSet = 1
group by  os.OptimizerRunSet, p.OptimizerRun, p.Board_ID
