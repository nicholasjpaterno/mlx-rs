extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf};

fn build_and_link_mlx_c() {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Set macOS deployment target for Metal/MLX compatibility
    // Uses MACOSX_DEPLOYMENT_TARGET env var if set, otherwise defaults to 14.0
    // This avoids linking errors with ___isPlatformVersionAtLeast
    // See: https://github.com/ml-explore/mlx/issues/1602
    #[cfg(target_os = "macos")]
    {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_string());
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
        // Ensure environment is set for compiler-rt symbols
        std::env::set_var("MACOSX_DEPLOYMENT_TARGET", &deployment_target);
        println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    }

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Set minimum macOS version for linker to match compiled objects
    // Must match CMAKE_OSX_DEPLOYMENT_TARGET to avoid ___isPlatformVersionAtLeast errors
    #[cfg(target_os = "macos")]
    {
        let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_string());
        println!("cargo:rustc-link-arg=-mmacosx-version-min={}", deployment_target);

        // Link against clang_rt.osx for ___isPlatformVersionAtLeast symbol
        // Rust uses -nodefaultlibs which prevents clang from adding this automatically
        // See: https://github.com/rust-lang/rust/issues/109717
        let clang_rt_paths = [
            "/Library/Developer/CommandLineTools/usr/lib/clang/17/lib/darwin",
            "/Library/Developer/CommandLineTools/usr/lib/clang/16/lib/darwin",
            "/Library/Developer/CommandLineTools/usr/lib/clang/15/lib/darwin",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/lib/darwin",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15/lib/darwin",
        ];

        for path in clang_rt_paths {
            let lib_path = std::path::Path::new(path).join("libclang_rt.osx.a");
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", path);
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
                break;
            }
        }
    }
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
