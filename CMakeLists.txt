cmake_minimum_required(VERSION 3.22)
project(pyin_pitch_detection VERSION 1.0.0)

option(PYIN_ENABLE_TESTS "Enable testing" OFF)

find_package(JUCE REQUIRED)

add_library(pyin_lib
    STATIC
        src/pyin.cpp)

target_compile_features(pyin_lib 
    PUBLIC cxx_std_14)

target_include_directories(pyin_lib
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/src)

target_link_libraries(pyin_lib
    PUBLIC
        juce::juce_dsp)

if(PYIN_ENABLE_TESTS)
    enable_testing()
    add_compile_options(-O3 -march=native) # for reproducibility
    add_subdirectory(tests)
endif()