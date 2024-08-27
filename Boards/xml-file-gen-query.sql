select b.Board_Name as boardname,
    t.Num_On_Board as tracknum,
    t.Length as length,
    t.Two_Deck_Length as twodecklength,
    
    e.Event, e.Start, e.End
from Board b
inner join Track t
on t.Board_ID = b.Board_ID
inner join
(select l.Board_ID, l.Track_ID, "ladder" as Event, l.Start, l.End
from Ladder l

union all

select c.Board_ID, c.Track_ID, "chute" as Event, c.Start, c.End
from Chute c)
 e

on e.Board_ID = b.Board_ID
and e.Track_ID = t.Track_ID




where b.Board_Name = 'Meg Bday Trial #1'