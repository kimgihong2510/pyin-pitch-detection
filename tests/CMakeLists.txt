# Sine Frequency Sweep
add_executable(test_sine_freq_sweep sine_freq_sweep.cpp)

target_link_libraries(test_sine_freq_sweep
    PRIVATE 
        pyin_lib)

# Enhanced Params Analysis        
add_executable(test_enhanced_params_analysis enhanced_params_analysis.cpp)

target_link_libraries(test_enhanced_params_analysis
    PRIVATE 
        pyin_lib)

target_compile_definitions(test_enhanced_params_analysis
    PRIVATE
        AUDIO_DATASET_DIR="${CMAKE_CURRENT_LIST_DIR}/MDB-melody-synth/audio_melody"
        ANNOTATION_DATASET_DIR="${CMAKE_CURRENT_LIST_DIR}/MDB-melody-synth/annotation_melody")

# Real Time Factor 
add_executable(test_real_time_factor real_time_factor.cpp)

target_link_libraries(test_real_time_factor
    PRIVATE 
        pyin_lib)

target_compile_definitions(test_real_time_factor
    PRIVATE
        AUDIO_DATASET_DIR="${CMAKE_CURRENT_LIST_DIR}/MDB-melody-synth/audio_melody")