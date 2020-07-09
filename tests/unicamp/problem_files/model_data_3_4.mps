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