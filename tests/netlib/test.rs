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
    let expected = -3.599_176_728_657_651e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_AGG2() {
    let result = solve("AGG2");
    let expected = -2.023_925_235_597_711e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_AGG3() {
    let result = solve("AGG3");
    let expected = 1.031_211_593_508_922_5e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_BANDM() {
    let result = solve("BANDM");
    let expected = -1.586_280_184_501_206_3e2; // Koch - The final Netlib-LP results
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
    let expected = 1.977_629_561_522_889_2e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_BNL2() {
    let result = solve("BNL2");
    let expected = 1.811_236_540_358_545_2e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_BOEING1() {
    let result = solve("BOEING1");
    let expected = -3.352_135_675_071_266_4e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_BOEING2() {
    let result = solve("BOEING2");
    let expected = -3.150_187_280_152_029e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_BORE3D() {
    let result = solve("BORE3D");
    let expected = 1.373_080_394_208_492_6e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_BRANDY() {
    let result = solve("BRANDY");
    let expected = 1.518_509_896_488_128_3e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
fn test_CAPRI() {
    let result = solve("CAPRI");
    let expected = 2.690_012_913_768_161e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-5));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_CYCLE() {
    let result = solve("CYCLE");
    let expected = -5.226_393_024_894_102; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_CZPROB() {
    let result = solve("CZPROB");
    let expected = 2.185_196_698_856_577_3e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-2));
}

#[test]
fn test_E226() {
    let result = solve("E226");
    let expected = -1.875_192_906_637_055e1; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_ETAMACRO() {
    let result = solve("ETAMACRO");
    let expected = -7.557_152_333_749_133e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
fn test_FINNIS() {
    let result = solve("FINNIS");
    let expected = 1.727_910_655_956_115_8e5; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_GREENBEA() {
    let result = solve("GREENBEA");
    let expected = -7.255_524_812_984_599e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-2));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_GREENBEB() {
    let result = solve("GREENBEB");
    let expected = -4.302_260_261_206_587e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_GFRD_PNC() {
    let result = solve("GFRD-PNC");
    let expected = 6.902_235_999_548_809e6; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
#[ignore = "Constraint on objective is not supported."]
fn test_GROW7() {
    let result = solve("GROW7");
    let expected = 4.778_781_181_471_15e7; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-1));
}

#[test]
fn test_ISRAEL() {
    let result = solve("ISRAEL");
    let expected = -8.966_448_218_630_457e5; // Koch - The final Netlib-LP results
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
    let expected = 3.206_197_290_643_158e2; // Koch - The final Netlib-LP results
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
    let expected = 9.042_969_538_007_915e2; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
fn test_SCSD1() {
    let result = solve("SCSD1");
    let expected = 8.666_666_674_333_365; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-7));
}

#[test]
#[ignore = "Too computationally intensive."]
fn test_SCFXM1() {
    let result = solve("SCFXM1");
    let expected = 1.841_675_902_834_894_4e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-3));
}

#[test]
fn test_SCORPION() {
    let result = solve("SCORPION");
    let expected = 1.878_124_822_738_106_6e3; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-6));
}

#[test]
fn test_SCTAP1() {
    let result = solve("SCTAP1");
    let expected = 0.141225e4; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

#[test]
#[ignore = "Too computationally intensive: only 619 pivots, but the exact values in the basis \
inverse grow to hundreds of digits, so a pivot costs seconds rather than microseconds."]
fn test_STAIR() {
    let result = solve("STAIR");
    let expected = -2.512_669_511_929_633e2; // Koch - The final Netlib-LP results
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
    let expected = -7.658_931_857_918_568e4; // Koch - The final Netlib-LP results
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
    let expected = 1.298_314_624_613_613_7e5; // Koch - The final Netlib-LP results
    assert!((result.objective_value - RB!(expected)).abs() < RB!(1e-4));
}

/// The multipliers of a real instance have to certify that its solution is optimal.
///
/// One instance rather than all of them: this is about the path from the simplex back to the rows
/// that were read, which is the same path for every instance.
#[test]
fn certificate_for_afiro() {
    crate::netlib::certify("AFIRO");
}

#[test]
fn certificate_for_sc50a() {
    crate::netlib::certify("SC50A");
}

/// Instances where the default configuration fixes a dominated column, and so cannot produce a
/// certificate. Running them again with only the reductions that keep one is what stops those
/// instances from being the ones nobody checks.
#[test]
fn certificate_for_sctap1() {
    crate::netlib::certify("SCTAP1");
}

#[test]
fn certificate_for_adlittle() {
    crate::netlib::certify("ADLITTLE");
}

#[test]
fn certificate_for_scorpion() {
    crate::netlib::certify("SCORPION");
}

#[test]
fn certificate_for_finnis() {
    crate::netlib::certify("FINNIS");
}

#[test]
fn certificate_for_scrs8() {
    crate::netlib::certify("SCRS8");
}

#[test]
fn certificate_for_czprob() {
    crate::netlib::certify("CZPROB");
}
