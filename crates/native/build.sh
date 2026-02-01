#!/bin/bash
set -e

# Nexus RAG Native Build Script
# Builds native libraries for iOS, Android, and Desktop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
OUTPUT_DIR="$SCRIPT_DIR/build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Rust is installed
check_rust() {
    if ! command -v rustc &> /dev/null; then
        log_error "Rust is not installed"
        exit 1
    fi
    log_info "Rust $(rustc --version)"
}

# Build for iOS
build_ios() {
    log_info "Building for iOS..."

    # Add targets
    rustup target add aarch64-apple-ios
    rustup target add x86_64-apple-ios  # Simulator
    rustup target add aarch64-apple-ios-sim  # M1 Simulator

    # Build for device (ARM64)
    log_info "Building for iOS device (aarch64)..."
    cargo build --release --target aarch64-apple-ios -p nexus-rag-native

    # Build for simulators
    log_info "Building for iOS simulator (x86_64)..."
    cargo build --release --target x86_64-apple-ios -p nexus-rag-native

    log_info "Building for iOS simulator (aarch64)..."
    cargo build --release --target aarch64-apple-ios-sim -p nexus-rag-native

    # Create XCFramework
    create_xcframework
}

create_xcframework() {
    log_info "Creating XCFramework..."

    mkdir -p "$OUTPUT_DIR/ios"

    xcodebuild -create-xcframework \
        -library "target/aarch64-apple-ios/release/libnexus_rag_native.a" \
        -headers "crates/native/include" \
        -library "target/x86_64-apple-ios/release/libnexus_rag_native.a" \
        -headers "crates/native/include" \
        -library "target/aarch64-apple-ios-sim/release/libnexus_rag_native.a" \
        -headers "crates/native/include" \
        -output "$OUTPUT_DIR/ios/NexusRAG.xcframework"

    log_info "XCFramework created at $OUTPUT_DIR/ios/NexusRAG.xcframework"
}

# Build for Android
build_android() {
    log_info "Building for Android..."

    # Check for NDK
    if [ -z "$ANDROID_NDK_HOME" ]; then
        log_error "ANDROID_NDK_HOME is not set"
        exit 1
    fi

    # Add targets
    rustup target add aarch64-linux-android
    rustup target add armv7-linux-androideabi
    rustup target add i686-linux-android
    rustup target add x86_64-linux-android

    # Build for each architecture
    log_info "Building for Android ARM64..."
    cargo build --release --target aarch64-linux-android -p nexus-rag-native

    log_info "Building for Android ARMv7..."
    cargo build --release --target armv7-linux-androideabi -p nexus-rag-native

    log_info "Building for Android x86_64..."
    cargo build --release --target x86_64-linux-android -p nexus-rag-native

    log_info "Building for Android i686..."
    cargo build --release --target i686-linux-android -p nexus-rag-native

    # Copy to Android jniLibs structure
    create_android_libs
}

create_android_libs() {
    log_info "Creating Android jniLibs structure..."

    mkdir -p "$OUTPUT_DIR/android/jniLibs/arm64-v8a"
    mkdir -p "$OUTPUT_DIR/android/jniLibs/armeabi-v7a"
    mkdir -p "$OUTPUT_DIR/android/jniLibs/x86_64"
    mkdir -p "$OUTPUT_DIR/android/jniLibs/x86"

    cp "target/aarch64-linux-android/release/libnexus_rag_native.so" \
       "$OUTPUT_DIR/android/jniLibs/arm64-v8a/"

    cp "target/armv7-linux-androideabi/release/libnexus_rag_native.so" \
       "$OUTPUT_DIR/android/jniLibs/armeabi-v7a/"

    cp "target/x86_64-linux-android/release/libnexus_rag_native.so" \
       "$OUTPUT_DIR/android/jniLibs/x86_64/"

    cp "target/i686-linux-android/release/libnexus_rag_native.so" \
       "$OUTPUT_DIR/android/jniLibs/x86/"

    log_info "Android libraries created at $OUTPUT_DIR/android/jniLibs"
}

# Build for Desktop
build_desktop() {
    log_info "Building for Desktop..."

    # Build for current platform
    cargo build --release -p nexus-rag-native

    mkdir -p "$OUTPUT_DIR/desktop"

    # Copy library
    if [[ "$OSTYPE" == "darwin"* ]]; then
        cp "target/release/libnexus_rag_native.dylib" "$OUTPUT_DIR/desktop/" || true
        cp "target/release/libnexus_rag_native.a" "$OUTPUT_DIR/desktop/"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cp "target/release/libnexus_rag_native.so" "$OUTPUT_DIR/desktop/"
        cp "target/release/libnexus_rag_native.a" "$OUTPUT_DIR/desktop/"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cp "target/release/nexus_rag_native.dll" "$OUTPUT_DIR/desktop/" || true
        cp "target/release/nexus_rag_native.lib" "$OUTPUT_DIR/desktop/" || true
    fi

    # Copy header
    cp "crates/native/include/nexus_rag.h" "$OUTPUT_DIR/desktop/"

    log_info "Desktop libraries created at $OUTPUT_DIR/desktop"
}

# Main
main() {
    check_rust

    mkdir -p "$OUTPUT_DIR"

    case "${1:-all}" in
        ios)
            build_ios
            ;;
        android)
            build_android
            ;;
        desktop)
            build_desktop
            ;;
        all)
            if [[ "$OSTYPE" == "darwin"* ]]; then
                build_ios
            fi

            if [ -n "$ANDROID_NDK_HOME" ]; then
                build_android
            else
                log_warn "Skipping Android build (ANDROID_NDK_HOME not set)"
            fi

            build_desktop
            ;;
        clean)
            log_info "Cleaning build artifacts..."
            rm -rf "$OUTPUT_DIR"
            cargo clean
            ;;
        *)
            echo "Usage: $0 {ios|android|desktop|all|clean}"
            exit 1
            ;;
    esac

    log_info "Build complete!"
}

main "$@"
