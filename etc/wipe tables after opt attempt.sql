BEGIN TRANSACTION;
delete from OptimizerRunResults;
delete from OptimizerRuns;
delete from OptimizerRunTestParams;
delete from EventHit;
END TRANSACTION;