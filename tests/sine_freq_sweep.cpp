#include "pyin.h"
#include "utils.h"

#include <iostream>
#include <vector>

using namespace pyin_pitch_detection;

// Test configuration
double kSampleRate = 44100.0;
float kMinFreq = 100.0f;
float kMaxFreq = 3000.0f;
float kErrorThresholds[] = {0.1f, 0.5f, 1.0f}; // Correct if the error is within the allowed semitone range.

struct Result {
    float estFreq;
    float trueFreq;
};

int main(void) {
    std::vector<Result> results;

    int frameSize = Pyin::getSmallestFrameSize(kSampleRate, kMinFreq);
    Pyin pitchDetector(kSampleRate, frameSize, kMinFreq, kMaxFreq);
    
    std::vector<float> input(frameSize);

    // frequency sweep; upward
    for(float trueFreq = kMinFreq; trueFreq < kMaxFreq; trueFreq += 1.0f) { // 1.0 Hz increment
        test_utils::setSineWave(kSampleRate, input, trueFreq);
        const float estFreq = pitchDetector.process(input.data()).freq;
        
        results.push_back(Result{estFreq, trueFreq});
    }
    
    // frequency sweep; downward
    for(float trueFreq = kMaxFreq; trueFreq < kMinFreq; trueFreq -= 1.0f) { // 1.0 Hz increment
        test_utils::setSineWave(kSampleRate, input, trueFreq);
        const float estFreq = pitchDetector.process(input.data()).freq;
        
        results.push_back(Result{estFreq, trueFreq});
    } 
 
    for(const auto errorThreshold: kErrorThresholds) {
        int correctCount = 0;
        for(const auto& result: results)
            if(test_utils::distanceInSemitone(result.estFreq, result.trueFreq) < errorThreshold)
                ++correctCount;

        float accuracy = static_cast<float>(correctCount) / static_cast<float>(results.size());
        std::cout << "Accuracy @ " << errorThreshold << " semitone threshold = " << accuracy * 100.0f << "%\n";
    }
}