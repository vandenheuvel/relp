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