cmake_minimum_required(VERSION 3.18)
project(sci2026_rerun LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Local opennn checkout + prebuilt slim static lib (same as IDC_benchmark).
set(OPENNN_ROOT "C:/Users/Artelnics/Documents/opennn" CACHE PATH "opennn checkout")
set(OPENNN_LIB  "${OPENNN_ROOT}/build-cmake/opennn/Release/opennn_core.lib" CACHE FILEPATH "prebuilt opennn lib")

if(NOT EXISTS "${OPENNN_LIB}")
    message(FATAL_ERROR "opennn_core.lib not found at ${OPENNN_LIB}")
endif()
if(NOT EXISTS "${OPENNN_ROOT}/eigen/Eigen/Core")
    message(FATAL_ERROR "Eigen headers not found at ${OPENNN_ROOT}/eigen")
endif()

find_package(OpenMP REQUIRED)

add_library(opennn STATIC IMPORTED GLOBAL)
set_target_properties(opennn PROPERTIES
    IMPORTED_LOCATION "${OPENNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENNN_ROOT}/opennn;${OPENNN_ROOT}/eigen")

function(add_idc_exe target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
    target_link_libraries(${target} PRIVATE opennn OpenMP::OpenMP_CXX)
    if(MSVC)
        # Drop /WHOLEARCHIVE (LNK1327 on the dev-refactor lib) — IDC drivers
        # instantiate ApproximationNetwork/ResponseOptimization directly.
        target_compile_options(${target} PRIVATE "/bigobj")
        target_compile_definitions(${target} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

add_idc_exe(run_idc_photo_wf3 cpp/run_idc_photo_wf3.cpp)
add_idc_exe(run_idc_yacht_mo2 cpp/run_idc_yacht_mo2.cpp)
add_idc_exe(parity_dump       cpp/parity_dump.cpp)
