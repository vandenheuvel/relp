************************************************************************
*
*  The data in this file represents the following problem:
*
*  Minimize or maximize Z = x1 + 2x2 + x3
*
*  Subject to:
*
*   2.0 <=   x1 +  x2         <= 4.0
*   0.0 <=      -  x2  +  x3  <= 0.0
*  -3.0 <=  -x1        -  x3  <=-3.0
*
*  where:
*
*   0.0 <= x1 <= 1.0
*   0.0 <= x2
*   0.0 <= x3
*
************************************************************************
NAME          NEXAMPLE
ROWS
 N  OBJ
 G  ROW01
 E  ROW02
 E  ROW03
COLUMNS
    COL01     OBJ                1.0
    COL01     ROW01              1.0   ROW03             -1.0
    COL02     OBJ                2.0   ROW01              1.0
    COL02     ROW02             -1.0
    COL03     OBJ                1.0   ROW02              1.0
    COL03     ROW03             -1.0
RHS
    RHS1      ROW01              2.0
    RHS1      ROW03             -3.0
RANGES
    RNG1      ROW01              2.0
BOUNDS
 UP BND1      COL01              1.0
ENDATA
