************************************************************************
*
*  This MPS file is intended for use with the EXDSCM2 and EXINIT sample
*  programs and contains four small LP models. EXINIT uses only the
*  first three. EXDSCM2 solves the first three models as minimization
*  problems and the optimal objective values obtained are used to build
*  the the objective function of the fourth problem (maximization).
*
*  The fourth problem is as follows:
*
*  Maximize Z = -(opt. obj. value from 1st problem)*x1 -
*               (opt. obj. value from 2nd problem)*x2 -
*               (opt. obj. value from 3rd problem)*x3 +  4.50x4
*
*  Subject to:
*
*         x1 + x2 + x3  = 12.0
*  4.0 <=      x2 + x3
*         x1 +      x3 <=  9.0
*  2.0 <= x1 + x2 + x3 <=  8.0
*
*  where:
*
*  0.0 <= x1
*  0.0 <= x2
*  0.0 <= x3
*         x4 = 500.0
*
************************************************************************
NAME          RAW1COST
ROWS
 N  COST
 G  SUP1COST
 G  SUP2COST
 G  SUP3COST
 L  PURITY
 E  AMOUNT
COLUMNS
    SUP1      COST               .20   SUP1COST           .20
    SUP1      PURITY             .08   AMOUNT            1.00
    SUP2      COST               .80   SUP2COST           .80
    SUP2      PURITY             .02   AMOUNT            1.00
    SUP3      COST               .30   SUP3COST           .30
    SUP3      PURITY             .04   AMOUNT            1.00
RHS
    RHS       SUP1COST         10.00
    RHS       AMOUNT          200.00   PURITY           10.00
BOUNDS
 UP BOUND     SUP2             75.00
 UP BOUND     SUP3            100.00
ENDATA
************************************************************************
NAME          RAW2COST
ROWS
 N  COST
 G  SUP1COST
 G  SUP2COST
 L  PURITY
 E  AMOUNT
COLUMNS
    SUP1      COST              3.60   SUP1COST          3.60
    SUP1      PURITY             .20   AMOUNT            1.00
    SUP2      COST              1.20   SUP2COST          1.20
    SUP2      PURITY             .40   AMOUNT            1.00
RHS
    RHS       AMOUNT          100.00   PURITY           35.00
ENDATA
************************************************************************
NAME          RAW3COST
ROWS
 N  COST
 G  SUP1COST
 G  SUP2COST
 L  PURITY
 E  AMOUNT
COLUMNS
    SUP1      COST              1.40   SUP1COST          1.40
    SUP1      PURITY             .01   AMOUNT            1.00
    SUP2      COST               .70   SUP2COST           .70
    SUP2      PURITY             .07   AMOUNT            1.00
RHS
    RHS       AMOUNT          250.00   PURITY           12.50
BOUNDS
 UP BOUND     SUP2            150.00
ENDATA
************************************************************************
NAME          PROFITS
ROWS
 N  PROFIT
 E  AMOUNT1
 G  AMOUNT2
 L  AMOUNT3
 L  AMOUNT4
COLUMNS
    RAW1      AMOUNT1           1.00
    RAW1      AMOUNT3           1.00   AMOUNT4           1.00
    RAW2      AMOUNT1           1.00
    RAW2      AMOUNT2           1.00   AMOUNT4           1.00
    RAW3      AMOUNT1           1.00
    RAW3      AMOUNT2           1.00   AMOUNT3           1.00
    PRODUCT   PROFIT            4.50
RHS
    RHS       AMOUNT1          12.00   AMOUNT2           4.00
    RHS       AMOUNT3           9.00   AMOUNT4           8.00
RANGES
    RANGE     AMOUNT4           6.00
BOUNDS
 FX BOUND     PRODUCT         500.00
ENDATA
