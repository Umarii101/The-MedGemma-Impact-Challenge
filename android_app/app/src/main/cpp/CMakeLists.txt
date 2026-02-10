cmake_minimum_required(VERSION 3.22.1)
project("medgemma-jni" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Force release-level optimization on EVERY source file.
#
# Problem: Gradle's assembleDebug sets CMAKE_BUILD_TYPE=Debug. The NDK
# toolchain populates CMAKE_C_FLAGS with Android-specific flags, but the
# Debug config provides NO optimization (-O0 implicit). The llama.cpp
# subdirectory calls project() which resets variable scopes, so our
# CMAKE_C_FLAGS CACHE FORCE does NOT propagate to ggml-base or llama targets.
#
# Solution: Use THREE independent mechanisms to guarantee -O3 everywhere:
#   1. add_compile_options() — directory property inherited by ALL subdirs
#   2. CMAKE_<LANG>_FLAGS_DEBUG — config-specific flags for Debug builds
#   3. CMAKE_<LANG>_FLAGS CACHE — belt-and-suspenders
# The LAST -O flag on the command line wins (GCC/Clang).
# ══════════════════════════════════════════════════════════════════════════════

# Mechanism 1: Directory-level compile options — propagates to all add_subdirectory() targets
add_compile_options(-O3 -DNDEBUG)

# Mechanism 2: Override Debug config flags (normally empty/-O0)
set(CMAKE_C_FLAGS_DEBUG   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)

# Mechanism 3: Append to global flags
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG" CACHE STRING "" FORCE)

# ── ARM architecture ─────────────────────────────────────────────────────────
# Snapdragon 8s Gen 3 → ARMv9-A with dotprod, i8mm, fp16.
# The NDK toolchain does NOT set CMAKE_SYSTEM_PROCESSOR, so ggml's
# ggml_get_system_arch() returns "UNKNOWN" and the ARM-optimized kernel
# files (arch/arm/quants.c, arch/arm/repack.cpp) are never compiled.
# Fix: set CMAKE_SYSTEM_PROCESSOR AND apply -march globally so that even
# ggml-base's ggml-quants.c gets dotprod/i8mm intrinsics.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_SYSTEM_PROCESSOR "aarch64")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm+fp16" CACHE STRING "" FORCE)
    # Global -march so ALL .c/.cpp files use dotprod/i8mm (not just ggml-cpu)
    add_compile_options(-march=armv8.2-a+dotprod+i8mm+fp16)
    message(STATUS "ARM: aarch64, armv8.2-a+dotprod+i8mm+fp16 (GLOBAL)")
endif()

# ── Locate llama.cpp source ──────────────────────────────────────────────────
if(NOT DEFINED LLAMA_CPP_DIR)
    set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../llama_cpp_repo")
endif()
get_filename_component(LLAMA_CPP_DIR "${LLAMA_CPP_DIR}" ABSOLUTE)
message(STATUS "llama.cpp directory: ${LLAMA_CPP_DIR}")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp source not found at: ${LLAMA_CPP_DIR}")
endif()

# ── ABI-specific tuning ─────────────────────────────────────────────────────
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)  # FetchContent fails with spaces in path

# ── llama.cpp build options ─────────────────────────────────────────────────
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          ON  CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE       OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} build-llama)

# ── JNI bridge library ─────────────────────────────────────────────────────
add_library(${CMAKE_PROJECT_NAME} SHARED medgemma_jni.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_CPP_DIR}
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
    llama
    common
    android
    log)
