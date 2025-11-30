# PYIN C++ implementation
Real-time C++ implementation of [PYIN] 
(Based on JUCE + C++ Standard Library only)

### Requirements
- C++ 14 or later
- CMake 3.22 or later
- JUCE Framework 8.0.7 or later
## Usage
### How to integrate into your project
1. Clone this repository into a dependency folder (e.g., `your_project/dep`)
```bash
$ mkdir dep
$ cd dep
$ git clone https://github.com/kimgihong2510/pyin-pitch-detection.git
```
2. Add the directory to your main `CMakeLists.txt` file
```CMake
add_subdirectory(dep/pyin-pitch-detection)

target_link_libraries(YourTarget PRIVATE pyin_lib)
```
### Code Example
```C++
#include "pyin.h"

float minFreq = 100.0f; // Minimum frequency to be estimated

// Note: getSmallestFrameSize() returns the minimum frame size that pyin can operate with,
// but using getDefaultFrameSize() is usually recommended for stable result. 
int frameSize = pyin_pitch_detection::Pyin::getSmallestFrameSize(sampleRate, minFreq); 
int frameSize = pyin_pitch_detection::Pyin::getDefaultFrameSize(sampleRate, minFreq); // RECOMMENDED
pyin_pitch_detection::Pyin pyin{};
pyin.prepareToPlay(sampleRate, frameSize);

std::vector<float> audioBlock(frameSize);
auto result = pyin.process(audioBlock.data());
float freq = result.freq; // Estimated f0 (in Hz)
bool isVoiced = result.isVoiced; // Estimated voiced/unvoiced state
```
### Code Example (Detailed Configuration)
This library allows you to configure parameters such as min frequency and max frequency to estimate, and more.
```C++
struct EnhancedParams {
    double voicedTrust = 0.5; // Prior probability that a frame is voiced
    double yinTrust = 0.99; // Weight indicating how strongly to trust the YIN-based likelihood
    TransitionParams transitionParams{}; // Transition Probabilities
};

auto enhancedParams = pyin_pitch_detection::Pyin::getDefaultEnhancedParams();

// modify enhancedParams...

pyin_pitch_detection::Pyin::pyin{};
pyin.prepareToPlay(sampleRate, frameSize, minFreq, maxFreq, enhancedParams);
```
## Tests
### Build Tests
1. Enable testing in the root `CMakeLists.txt`
```CMake
option(PYIN_ENABLE_TESTS "Enable testing" OFF) 
-> option(PYIN_ENABLE_TESTS "Enable testing" ON)
```
2. Build the tests in `build/`
```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
Now, test executables will appear in `build/tests/`.    

### Test Results
#### 1. test_sine_freq_sweep
Measures pitch accuracy on a sine sweep. 
```bash
$ ./build/tests/test_sine_freq_sweep
Accuracy @ 0.1 semitone threshold = 47.1724%
Accuracy @ 0.5 semitone threshold = 89.5172%
Accuracy @ 1 semitone threshold = 94.8276%
```
#### 2.test_enhanced_params_analysis
Evaluates
- pitch accuracy
- voiced -> unvoiced errors (misclassifying voiced frame to unvoiced)
- unvoiced -> voiced errors (misclassifying unvoiced frame to voiced)  
  
under different EnhancedParams configurations on [MDB-melody-synth] dataset.   
The results are stored in: `tests/pyin_performance_results.csv`   

Before running this test, place the `MDB-melody-synth` folder inside `tests/`, then run:
```bash
$ ./build/tests/test_enhanced_params_analysis <voicedTrust> <yinTrust> <voicedToUnvoicedProb> <unvoicedToVoicedProb>
```
For easier parameter exploration, an automated Python script is provided: `tests/automated_enhanced_params_analysis.py`

#### 3. test_real_time_factor
Measures the real-time factor on [MDB-melody-synth] dataset.
```bash
$ ./build/tests/test_real_time_factor 
start experiment
finished experiment
=====================================================
=====================================================
result: 
Total Audio Duration: 11484.4s
Total Processing Time: 113s
Real Time Factor: 0.00983945
```
(The test results were obtained on Clang 17, MacBook Pro (M2), 16 GB RAM, macOS 15.5.)

## TODO
- This pYIN implementation performs poorly at low frequencies. Further improvements are needed.
- Compare performance with the original PYIN implementation.

[PYIN]: https://ieeexplore.ieee.org/document/6853678
[MDB-melody-synth]: https://synthdatasets.weebly.com/mdb-melody-synth.html

