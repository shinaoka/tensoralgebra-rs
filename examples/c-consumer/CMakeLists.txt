# A CMake project that consumes the TAPP surface, the way a C or C++ codebase
# actually would.
#
#   cmake -S examples/c-consumer -B build/c-consumer -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/c-consumer
#   ./build/c-consumer/tapp-c-consumer
#
# Two ways to get the library are supported, because they suit different sites:
#
#   TAPP_USE_CORROSION=ON  (default)
#     Corrosion drives cargo from CMake and imports the crate as a normal CMake
#     target, so `target_link_libraries` is all the consumer writes. It works
#     out the platform link line — on Linux a Rust staticlib still needs
#     -lpthread -ldl -lm, and corrosion knows that so you do not have to.
#     Requires a Rust toolchain at *build* time (not at run time), and network
#     access on the first configure to fetch corrosion itself.
#
#   TAPP_USE_CORROSION=OFF
#     Link a library that is already built, via TAPP_LIBRARY_DIR. This is the
#     path for a site that builds the Rust half separately — or offline, or
#     from a vendored tarball — and for anyone who would rather not have CMake
#     invoke cargo at all.
#
# Nothing here needs a Rust toolchain at run time, and nothing crosses the C++
# ABI: the boundary is C, so there is no name mangling, no exception
# propagation and no libstdc++ version coupling.

cmake_minimum_required(VERSION 3.22)
project(tapp-c-consumer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TAPP_USE_CORROSION "Build the Rust library via corrosion" ON)

# The crate root, i.e. the repository root two levels up from this file.
get_filename_component(TAPP_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TAPP_CRATE_DIR "${TAPP_REPO_ROOT}/crates/tensorprimitives-tapp")

add_executable(tapp-c-consumer main.c)
target_include_directories(tapp-c-consumer PRIVATE "${TAPP_CRATE_DIR}/include")

if(TAPP_USE_CORROSION)
    include(FetchContent)
    FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5.1
    )
    FetchContent_MakeAvailable(Corrosion)

    # `cargo metadata` on the workspace manifest, then import just the crate
    # that exports the C ABI. CRATES= keeps the unpublished bench harness --
    # whose build script wants TBLIS_ROOT -- out of the build entirely.
    corrosion_import_crate(
        MANIFEST_PATH "${TAPP_REPO_ROOT}/Cargo.toml"
        CRATES tensorprimitives-tapp
    )

    target_link_libraries(tapp-c-consumer PRIVATE tensorprimitives_tapp)
else()
    if(NOT DEFINED TAPP_LIBRARY_DIR)
        set(TAPP_LIBRARY_DIR "${TAPP_REPO_ROOT}/target/release")
    endif()

    find_library(TAPP_LIBRARY
        NAMES tensorprimitives_tapp
        HINTS "${TAPP_LIBRARY_DIR}"
        REQUIRED
        NO_DEFAULT_PATH
    )
    message(STATUS "Using prebuilt TAPP library: ${TAPP_LIBRARY}")

    target_link_libraries(tapp-c-consumer PRIVATE "${TAPP_LIBRARY}")

    # A Rust *staticlib* bundles the standard library and needs these; a cdylib
    # does not, but asking for them is harmless either way. Threads are needed
    # regardless, since the engine's optional threading uses std::thread.
    find_package(Threads REQUIRED)
    target_link_libraries(tapp-c-consumer PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)

    # Let the built binary find a cdylib without LD_LIBRARY_PATH being set.
    set_target_properties(tapp-c-consumer PROPERTIES
        BUILD_RPATH "${TAPP_LIBRARY_DIR}"
    )
endif()

enable_testing()
add_test(NAME c-consumer COMMAND tapp-c-consumer)
