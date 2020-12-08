NAME          Testprob
*
*  Minimize:
*
*      x1 + 4 * x2 + 9 * x3
*
*  Subject to:
*
*      x1     + x2          <=  5
*      x1              + x3 >= 10
*             - x2     + x3 =   7
*
*  With:
*
*    0 <= x1 <= 4
*   -1 <= x2 <= 1
*         x3 unrestricted.
*
*  Tableau:
*
*            X1      X2     X3    |  RHS1
*       +--------------------------------
*  COST |   1.0     4.0    9.0    |   0.0
*  LIM1 |   1.0     1.0    0.0    |   5.0
*  LIM2 |   1.0     0.0    1.0    |  10.0
*  EQN  |   0.0    -1.0    1.0    |   7.0
*  -----+-------------------------+------
*  BND1 |   4.0    -1.0    1.0    | empty
*
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  EQN
COLUMNS
    X1        COST               1.0   LIM1               1.0
    X1        LIM2               1.0
    X2        COST               4.0   LIM1               1.0
    X2        EQN               -1.0
    X3        COST               9.0   LIM2               1.0
    X3        EQN                1.0
RHS
    RHS1      LIM1               5.0   LIM2              10.0
    RHS1      EQN                7.0
BOUNDS
 UP BND1      X1                 4.0
 LO BND1      X2                -1.0
 UP BND1      X2                 1.0
 FR BND1      X3
ENDATA