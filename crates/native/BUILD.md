# Native Build Guide

Instructions for building Nexus RAG native libraries for iOS, Android, and Desktop platforms.

## Prerequisites

### All Platforms
- Rust (1.70+): https://rustup.rs/
- Git

### iOS (macOS only)
- Xcode 14+
- Xcode Command Line Tools: `xcode-select --install`
- iOS targets: `rustup target add aarch64-apple-ios x86_64-apple-ios aarch64-apple-ios-sim`

### Android
- Android NDK 25+: https://developer.android.com/ndk
- Set `ANDROID_NDK_HOME` environment variable
- Android targets:
  ```bash
  rustup target add aarch64-linux-android
  rustup target add armv7-linux-androideabi
  rustup target add i686-linux-android
  rustup target add x86_64-linux-android
  ```
- cargo-ndk: `cargo install cargo-ndk`

### Desktop
- No additional requirements (uses host toolchain)

## Quick Start

```bash
# Build for all platforms
./crates/native/build.sh all

# Build for specific platform
./crates/native/build.sh ios
./crates/native/build.sh android
./crates/native/build.sh desktop

# Clean build artifacts
./crates/native/build.sh clean
```

## Platform-Specific Instructions

### iOS

**Outputs:**
- `build/ios/NexusRAG.xcframework` - XCFramework for Xcode integration

**Manual Build:**
```bash
# Device (ARM64)
cargo build --release --target aarch64-apple-ios -p nexus-rag-native

# Simulator (Intel)
cargo build --release --target x86_64-apple-ios -p nexus-rag-native

# Simulator (Apple Silicon)
cargo build --release --target aarch64-apple-ios-sim -p nexus-rag-native
```

**Xcode Integration:**
1. Drag `NexusRAG.xcframework` into Xcode project
2. Add to "Frameworks, Libraries, and Embedded Content"
3. Import header: `#include "nexus_rag.h"`

### Android

**Outputs:**
- `build/android/jniLibs/` - JNI libraries for all architectures
  - `arm64-v8a/libnexus_rag_native.so` (ARM64)
  - `armeabi-v7a/libnexus_rag_native.so` (ARMv7)
  - `x86_64/libnexus_rag_native.so` (x86_64)
  - `x86/libnexus_rag_native.so` (x86)

**Manual Build:**
```bash
cargo build --release --target aarch64-linux-android -p nexus-rag-native
cargo build --release --target armv7-linux-androideabi -p nexus-rag-native
cargo build --release --target x86_64-linux-android -p nexus-rag-native
cargo build --release --target i686-linux-android -p nexus-rag-native
```

**Android Studio Integration:**
1. Copy `build/android/jniLibs` to `app/src/main/jniLibs`
2. Create JNI wrapper (see examples/android)

### Desktop

**Outputs:**
- `build/desktop/libnexus_rag_native.{a,so,dylib,dll}` - Static/dynamic libraries
- `build/desktop/nexus_rag.h` - C header

**Manual Build:**
```bash
cargo build --release -p nexus-rag-native
```

**Linking:**
```c
// Compile with:
// gcc -o app app.c -L./build/desktop -lnexus_rag_native -lm
#include "nexus_rag.h"

int main() {
    RagHandle* rag = rag_create(384, 1);
    // ... use RAG ...
    rag_destroy(rag);
    return 0;
}
```

## Cargo Configuration

Create `.cargo/config.toml` for Android NDK:

```toml
[target.aarch64-linux-android]
ar = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang"

[target.armv7-linux-androideabi]
ar = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi30-clang"

[target.i686-linux-android]
ar = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android30-clang"

[target.x86_64-linux-android]
ar = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android30-clang"
```

## Troubleshooting

### iOS: "library not found"
- Ensure all targets are installed: `rustup target list | grep apple`
- Clean and rebuild: `cargo clean && ./build.sh ios`

### Android: NDK not found
- Set ANDROID_NDK_HOME: `export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/25.2.9519653`
- Verify: `echo $ANDROID_NDK_HOME`

### Desktop: Missing dependencies
- Install system libraries:
  - Ubuntu: `sudo apt install build-essential`
  - macOS: `xcode-select --install`
  - Windows: Install Visual Studio Build Tools

## File Sizes

Approximate sizes (release builds with default features):

| Platform | Architecture | Size (compressed) |
|----------|--------------|-------------------|
| iOS | ARM64 | ~500 KB |
| iOS | x86_64 | ~600 KB |
| Android | ARM64 | ~550 KB |
| Android | ARMv7 | ~500 KB |
| Desktop | x86_64 | ~550 KB |

## Performance

Native builds are optimized for:
- Binary size (link-time optimization)
- Runtime speed (CPU-specific optimizations)
- Memory efficiency

## Next Steps

- See `examples/ios` for Swift integration
- See `examples/android` for Kotlin/Java integration
- See `examples/desktop` for C/C++ examples
