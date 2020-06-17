*************************************************************************
*
*  The data in this file represents the following problem:
*
*  Minimize or maximize Z = (.03 + .09*lambda)x1 + .08x2 +
*                           (.17 + .25*lambda)x3 + .12x4 +
*                            .15x5 + .21x6 + .38x7
*
*  Subject to:
*
*            x1 +    x2 +    x3 +    x4 +    x5 +    x6 +    x7  = 2000.0
*         .15x1 + .04x2 + .02x3 + .04x4 + .02x5 + .01x6 + .07x7 <=   60.0
*         .02x1 + .04x2 + .01x3 + .02x4 + .02x5                 <=   40.0
*   a <=  .03x1 + .05x2 + .08x3 + .02x4 + .06x5 + .01x6         <=    b
*         .02x1 + .03x2                 + .01x5                 <=   30.0
* 1500.0<=.30x1 + .75x2 + .80x3 + .75x4 + .80x5 + .97x6
*   c <=  .50x1 + .06x2 + .08x3 + .12x4 + .02x5 + .01x6 + .97x7 <=    d
*
*
*  where:
*
*  a = ((100.0 +  60.0*lambda) - (100.0 +  40.0*lambda))
*    = (   0.0 +  20.0*lambda)
*  b = ( 100.0 +  60.0*lambda)
*  c = ( 300.0 + 100.0*lambda)
*  d = ((300.0 +  30.0*lambda) + ( 50.0 + 100.0*lambda))
*    = ( 350.0 + 130.0*lambda)
*
*    0.0 <= x1 <=  200.0
*    0.0 <= x2 <=  750.0
*  400.0 <= x3 <=  800.0
*  100.0 <= x4 <= (700.0  +  200.0*lambda)
*    0.0 <= x5 <= (1500.0 + 1800.0*lambda)
*    0.0 <= x6
*    0.0 <= x7
*
*  The parametric change vectors are named as follows:
*
*  CHANGOBJ (objective function)
*  CHANGRHS (right-hand side)
*  CHANGRNG (ranges)
*  CHANGBND (bounds)
*
*  If parametrics are not used, the linear problem is as above with
*  lambda set to zero.
*
*************************************************************************
NAME          SPMETALS
ROWS
 N  VALUE
 N  CHANGOBJ
 E  YIELD
 L  FE
 L  MN
 L  CU
 L  MG
 G  AL
 G  SI
COLUMNS
    BIN1      VALUE             .03    YIELD         1.00
    BIN1      FE                .15    MN             .02
    BIN1      CU                .03    MG             .02
    BIN1      AL                .30    SI              .5
    BIN1      CHANGOBJ          .09
    BIN2      VALUE             .08    YIELD         1.00
    BIN2      FE                .04    MN             .04
    BIN2      CU                .05    MG             .03
    BIN2      AL                .75    SI             .06
    BIN3      VALUE             .17    YIELD         1.00
    BIN3      FE                .02    MN             .01
    BIN3      CU                .08    AL             .80
    BIN3      SI                .08
    BIN3      CHANGOBJ          .25
    BIN4      VALUE             .12    YIELD         1.00
    BIN4      FE                .04    MN             .02
    BIN4      CU                .02    AL             .75
    BIN4      SI                .12
    BIN5      VALUE             .15    YIELD         1.00
    BIN5      FE                .02    MN             .02
    BIN5      CU                .06    MG             .01
    BIN5      AL                .80    SI             .02
    ALUM      VALUE             .21    YIELD         1.00
    ALUM      FE                .01    CU             .01
    ALUM      AL                .97    SI             .01
    SILICON   VALUE             .38    YIELD         1.00
    SILICON   FE                .03    SI             .97
RHS
    RHS       YIELD           2000.    FE             60.
    RHS       CU               100.    MN             40.
    RHS       MG                30.    AL            1500.
    RHS       SI               300.
    CHANGRHS  SI               100.
    CHANGRHS  CU                60.
RANGES
    RNG       SI                50.
    CHANGRNG  SI                30.
    CHANGRNG  CU                40.
BOUNDS
 UP BNN       BIN1            200.
 UP BNN       BIN2            750.
 LO BNN       BIN3            400.
 UP BNN       BIN3            800.
 LO BNN       BIN4            100.
 UP BNN       BIN4            700.
 UP BNN       BIN5            1500.
 UP CHANGBND  BIN4            200.
 UP CHANGBND  BIN5            1800.
ENDATA
