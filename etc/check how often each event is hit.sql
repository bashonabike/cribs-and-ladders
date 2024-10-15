                     SELECT
OptimizerRun,                     
 Track_ID,
                            StartHole,
                            EndHole,
                            Disp
                            , count(1)
                            from EventHit
                            
where OptimizerRun = 0
                            group by
                            OptimizerRun,
                            Track_ID,
                            StartHole,
                            EndHole,
                            Disp
                            order by 
                            OptimizerRun,
                            Track_ID,
                            StartHole