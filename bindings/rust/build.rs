fn main(){
    println!("cargo:rustc-link-search=native=../../build");
    println!("cargo:rustc-link-lib=static=dl_capi");
    println!("cargo:rustc-link-lib=dylib=c++");
}