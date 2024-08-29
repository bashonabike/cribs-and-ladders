select op1.Result, op2.OptimizerRun, op2.ResultValue - op1.ResultValue, (op2.ResultValue - op1.ResultValue)/op1.ResultValue
from OptimizerRunResults op1
inner join OptimizerRunResults op2
on op1.OptimizerRunSet = op2.OptimizerRunSet
and op1.Board_ID = op2.Board_ID
and op1.OptimizerRun + 1 = op2.OptimizerRun
and op1.Result = op2.Result
WHERE op1.ResultValue <> op2.ResultValue 
ORDER BY op2.OptimizerRunSet, op1.Result