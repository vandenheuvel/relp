************************************************************************
*
*  The data in this file represents the following problem:
*
*  Minimize or maximize Z = x1 + 2x5 - x8
*
*  Subject to:
*
*  2.5 <=   3x1 +  x2          - 2x4  - x5              -    x8
*                 2x2 + 1.1x3                                   <=  2.1
*                          x3              + x6                  =  4.0
*  1.8 <=                      2.8x4             -1.2x7         <=  5.0
*  3.0 <= 5.6x1                       + x5              + 1.9x8 <= 15.0
*
*  where:
*
*  2.5 <= x1
*    0 <= x2 <= 4.1
*    0 <= x3
*    0 <= x4
*  0.5 <= x5 <= 4.0
*    0 <= x6
*    0 <= x7
*    0 <= x8 <= 4.3
*
*  The problem is then revised in the following way:
*
*  1) Since ROW04 will have a bound changed, it is declared in the
*     ROW MODIFY section with its type unchanged.
*  2) The cost of COL01 is changed to 2.0.
*  3) COL07 is deleted.
*  4) A new column, COL77, is added to the problem.  It has a coefficient
*     of -1.5 in ROW04.
*  5) The lower bound of ROW04 is changed to 0.8.
*  6) The lower bound of COL77 is changed to 1.0.
*
************************************************************************
NAME          EXAMPLE
ROWS
 N  OBJ
 G  ROW01
 L  ROW02
 E  ROW03
 G  ROW04
 L  ROW05
COLUMNS
    COL01     OBJ                1.0
    COL01     ROW01              3.0   ROW05              5.6
    COL02     ROW01              1.0   ROW02              2.0
    COL03     ROW02              1.1   ROW03              1.0
    COL04     ROW01             -2.0   ROW04              2.8
    COL05     OBJ                2.0
    COL05     ROW01             -1.0   ROW05              1.0
    COL06     ROW03              1.0
    COL07     ROW04             -1.2
    COL08     OBJ               -1.0
    COL08     ROW01             -1.0   ROW05              1.9
RHS
    RHS1      ROW01              2.5
    RHS1      ROW02              2.1
    RHS1      ROW03              4.0
    RHS1      ROW04              1.8
    RHS1      ROW05             15.0
RANGES
    RNG1      ROW04              3.2
    RNG1      ROW05             12.0
BOUNDS
 LO BND1      COL01              2.5
 UP BND1      COL02              4.1
 LO BND1      COL05              0.5
 UP BND1      COL05              4.0
 UP BND1      COL08              4.3
ENDATA
NAME          EXAMPLE
ROWS
MODIFY
 G  ROW04
COLUMNS
MODIFY
    COL01     OBJ                2.0
DELETE
    COL07
AFTER         COL06
    COL77     ROW04             -1.5
RHS
MODIFY
    RHS1      ROW04              0.8
BOUNDS
MODIFY
 LO BND1      COL77              1.0
ENDATA
