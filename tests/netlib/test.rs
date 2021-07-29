use relp_num::Abs;
use relp_num::RB;

use crate::netlib::solve;

#[test]
#[ignore = "Too computationally intensive."]
fn test_25FV47() {
    let result = solve("25FV47");
    let expected = 5.5018459e+03; // Gurobi
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
#[ignore = "Too computationally intensive, perhaps underflows in `compute_full_solution_with_reduced_solution`."]
fn test_80BAU3B() {
    let result = solve("80BAU3B");
    let expected = 9.872241924e+05; // Gurobi
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5)); // Gurobi
}


#[test]
fn test_ADLITTLE() {
    let result = solve("ADLITTLE");
    let expected = 2.254949632e+05; // Gurobi
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3)); // Gurobi
}

#[test]
fn test_AFIRO() {
    let result = solve("AFIRO");
    let expected = -464.75314; // Coin LP 1.17.6
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

/// Degeneracy
#[test]
fn test_AGG() {
    let result = solve("AGG");
    let expected = -0.35991767286576506712640824319636e8; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_AGG2() {
    let result = solve("AGG2");
    let expected = -0.20239252355977109024317661926133e8; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_AGG3() {
    let result = solve("AGG3");
    let expected = 0.10312115935089225579061058796215e8; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_BANDM() {
    let result = solve("BANDM");
    let expected = -0.15862801845012064052174123768736e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_BEACONFD() {
    let result = solve("BEACONFD");
    let expected = 0.335924858072e5; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_BLEND() {
    let result = solve("BLEND");
    let expected = -30.81215; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_BNL1() {
    let result = solve("BNL1");
    let expected = 0.19776295615228892439564398331821e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_BNL2() {
    let result = solve("BNL2");
    let expected = 0.1811236540358545170448413697691e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_BOEING1() {
    let result = solve("BOEING1");
    let expected = -0.3352135675071266218429697314682e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_BOEING2() {
    let result = solve("BOEING2");
    let expected = -0.31501872801520287870462195913263e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_BORE3D() {
    let result = solve("BORE3D");
    let expected = 0.13730803942084927215581987251301e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_BRANDY() {
    let result = solve("BRANDY");
    let expected = 0.15185098964881283835426751550618e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
fn test_CAPRI() {
    let result = solve("CAPRI");
    let expected = 0.26900129137681610087717280693754e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_CYCLE() {
    let result = solve("CYCLE");
    let expected = -0.52263930248941017172447233836217e1; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_CZPROB() {
    let result = solve("CZPROB");
    let expected = 0.21851966988565774858951155947191e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-2));
}

#[test]
fn test_E226() {
    let result = solve("E226");
    let expected = -0.18751929066370549102605687681285e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_ETAMACRO() {
    let result = solve("ETAMACRO");
    let expected = -0.7557152333749133350792583667773e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_FINNIS() {
    let result = solve("FINNIS");
    let expected = 0.17279106559561159432297900375543e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_GREENBEA() {
    let result = solve("GREENBEA");
    let expected = -0.72555248129845987457557870574845e8; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-2));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_GREENBEB() {
    let result = solve("GREENBEB");
    let expected = -0.43022602612065867539213672544432e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_GFRD_PNC() {
    let result = solve("GFRD-PNC");
    let expected = 0.69022359995488088295415596232193e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Constraint on objective is not supported."]
fn test_GROW7() {
    let result = solve("GROW7");
    let expected = 0.47787811814711502616766956242865e8; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-1));
}

#[test]
fn test_ISRAEL() {
    let result = solve("ISRAEL");
    let expected = -0.89664482186304572966200464196045e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_KB2() {
    let result = solve("KB2");
    let expected = -1.749900130e+03; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_LOTFI() {
    let result = solve("LOTFI");
    let expected = -0.2526470606188e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-8));
}

#[test]
#[ignore = "Could be cycling."]
fn test_MODSZK1() {
    let result = solve("MODSZK1");
    let expected = 0.32061972906431580494333823530763e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_SC50A() {
    let result = solve("SC50A");
    let expected = -6.457507706e+01; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
fn test_SC50B() {
    let result = solve("SC50B");
    let expected = -70; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-8));
}

#[test]
fn test_SC105() {
    let result = solve("SC105");
    let expected = -5.220206121e+01; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-8));
}

#[test]
fn test_SC205() {
    let result = solve("SC205");
    let expected = -5.220206121e+01; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-8));
}

#[test]
fn test_SCAGR7() {
    let result = solve("SCAGR7");
    let expected = -2.331389824e+06; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_SCRS8() {
    let result = solve("SCRS8");
    let expected = 0.90429695380079143579923107948844e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_SCSD1() {
    let result = solve("SCSD1");
    let expected = 0.86666666743333647292533502995263e1; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_SCFXM1() {
    let result = solve("SCFXM1");
    let expected = 0.18416759028348943683579089143655e5; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_SCORPION() {
    let result = solve("SCORPION");
    let expected = 0.18781248227381066296479411763586e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_SCTAP1() {
    let result = solve("SCTAP1");
    let expected = 0.141225e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
#[ignore = "Could be cycling."]
fn test_STAIR() {
    let result = solve("STAIR");
    let expected = -0.25126695119296330352803637106304e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_STANDATA() {
    let result = solve("STANDATA");
    let expected = 0.12576995e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_STANDMPS() {
    let result = solve("STANDMPS");
    let expected = 0.14060175e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_STOCFOR1() {
    let result = solve("STOCFOR1");
    let expected = -4.113197622e+04; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_SHARE1B() {
    let result = solve("SHARE1B");
    let expected = -0.7658931857918568112797274346007e5; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
fn test_SHARE2B() {
    let result = solve("SHARE2B");
    let expected = -4.157322407e+02; // GLPK 4.65
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_RECIPELP() {
    let result = solve("RECIPELP");
    let expected = -0.266616e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_VTP_BASE() {
    let result = solve("VTP-BASE");
    let expected = 0.1298314624613613657395984384889e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}
