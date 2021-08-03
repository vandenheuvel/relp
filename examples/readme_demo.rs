fn main() {
    let path = std::path::Path::new("my_program.mps");
    let my_program = relp::io::import::<relp_num::RationalBig>(path);
}
