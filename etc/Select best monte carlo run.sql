WITH results as (
SELECT
    os.OptimizerRunSet,
    o.OptimizerRun,
    r.Board_ID,
    IFNULL(SUM(CASE WHEN Result = 'balance' THEN r.ResultValue END), 0) AS balance,
    IFNULL(SUM(CASE WHEN Result = 'energy_curvefit' THEN r.ResultValue END), 0) AS energy_curvefit,
    IFNULL(SUM(CASE WHEN Result = 'eventSpacingHist_curvefit' THEN r.ResultValue END), 0) AS eventSpacingHist_curvefit,
    IFNULL(SUM(CASE WHEN Result = 'eventsHitLengthDistribution_curvefit' THEN r.ResultValue END), 0) AS eventsHitLengthDistribution_curvefit,
    IFNULL(SUM(CASE WHEN Result = 'eventsOverTime_curvefit' THEN r.ResultValue END), 0) AS eventsOverTime_curvefit,
    IFNULL(SUM(CASE WHEN Result = 'gamelength' THEN r.ResultValue END), 0) AS gamelength,
    IFNULL(SUM(CASE WHEN Result = 'multis' THEN r.ResultValue END), 0) AS multis,
    IFNULL(SUM(CASE WHEN Result = 'orthos' THEN r.ResultValue END), 0) AS orthos,
    IFNULL(SUM(CASE WHEN Result = 'repeats' THEN r.ResultValue END), 0) AS repeats,
    IFNULL(SUM(CASE WHEN Result = 'soexcite' THEN r.ResultValue END), 0) AS soexcite,
    IFNULL(SUM(CASE WHEN Result = 'twohits' THEN r.ResultValue END), 0) AS twohits,
    IFNULL(SUM(CASE WHEN Result = 'velocity_curvefit' THEN r.ResultValue END), 0) AS velocity_curvefit
FROM
    OptimizerRunResults r
    INNER JOIN OptimizerRuns o ON o.OptimizerRun = r.OptimizerRun
    INNER JOIN OptimizerRunSets os ON os.OptimizerRunSet = o.OptimizerRunSet
WHERE
    os.OptimizerRunSet = 1
GROUP BY
    os.OptimizerRunSet,
    o.OptimizerRun,
    r.Board_ID),
weighted as (    
    
select OptimizerRun,
balance*30 + 
energy_curvefit*1 + 
eventSpacingHist_curvefit*0.00005 + 
eventsHitLengthDistribution_curvefit*0.000000002 + 
eventsOverTime_curvefit*1 + 
gamelength*2 + 
multis*1 + 
orthos*0.5 + 
repeats*20 + 
soexcite*1 + 
twohits*0.5 + 
velocity_curvefit*1 as weight


 from results)
select * from weighted
order by weight
