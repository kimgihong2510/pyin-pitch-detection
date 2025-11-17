#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <vector>

namespace test_utils{

constexpr float PI = 3.14159265358979323846f;
void setSineWave(double sampleRate, std::vector<float>& input, float freq) {
    int frameSize = input.size();    
    
    float pos = 0.0f;
    for(int j=0; j<frameSize; j++) {
        input[j] = std::sin(pos);
        pos += 2*PI * freq / sampleRate;
    } 
}

float distanceInSemitone(float freq1, float freq2) {
    assert(freq1 > 0.0f && freq2 > 0.0f);
    return std::abs(std::log2(freq1/freq2)) * 12.0f;
}

} // namespace teset_utils