// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=gbm");

    // Shim that exposes the ffmpeg AVVulkanDeviceContext / AVVulkanFramesContext
    // / AVVkFrame fields to Rust. See src/ffv1_vk_shim.c for the rationale.
    println!("cargo:rerun-if-changed=src/ffv1_vk_shim.c");

    // Pick up ffmpeg / vulkan include paths via pkg-config when available.
    let mut build = cc::Build::new();
    build.file("src/ffv1_vk_shim.c");

    if let Ok(out) = Command::new("pkg-config")
        .args(["--cflags-only-I", "libavutil", "vulkan"])
        .output()
    {
        if out.status.success() {
            let cflags = String::from_utf8_lossy(&out.stdout);
            for tok in cflags.split_whitespace() {
                if let Some(path) = tok.strip_prefix("-I") {
                    build.include(path);
                }
            }
        }
    }

    // Fallback include paths if pkg-config is unavailable.
    build.include("/usr/include");

    // Silence harmless warnings from system ffmpeg headers — these are not
    // our code.
    build.warnings(false).extra_warnings(false);

    build.compile("waymux_ffv1_vk_shim");

    println!("cargo:rustc-link-lib=avutil");
}
