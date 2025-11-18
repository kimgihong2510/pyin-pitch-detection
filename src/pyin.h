#pragma once

#include <juce_dsp/juce_dsp.h>
#include <vector>

static constexpr float kDefaultMinFreq = 100.0f;
static constexpr float kDefaultMaxFreq = 3000.0f;

namespace pyin_pitch_detection {

struct TransitionParams { 
    double voicedToVoiced = 0.99;
    double unvoicedToVoiced = 0.01;
    double voicedToUnvoiced = 0.01;
    double unvoicedToUnvoiced = 0.99;
};
    
struct EnhancedParams {
    double voicedTrust = 0.5;
    double yinTrust = 0.99;
    TransitionParams transitionParams{};
};

struct PyinResult {
    float freq;       
    bool isVoiced;      
};

class Pyin {
public:
    enum class VoiceState {
        UNVOICED = 0,
        VOICED = 1
    };

    static int getSmallestFrameSize(double sampleRate, float minFreq = kDefaultMinFreq);
    static int getDefaultFrameSize(double sampleRate, float minFreq = kDefaultMinFreq);
    static EnhancedParams getDefaultEnhancedParams();

    Pyin(double sampleRate, int frameSize, float minFreq, float maxFreq, EnhancedParams enhancedParams);
    Pyin(double sampleRate, int frameSize) : Pyin(sampleRate, frameSize, kDefaultMinFreq, kDefaultMaxFreq, EnhancedParams{}) {};
    Pyin(double sampleRate, int frameSize, float minFreq, float maxFreq) : Pyin(sampleRate, frameSize, minFreq, maxFreq, EnhancedParams{}) {};;
    Pyin(double sampleRate, int frameSize, EnhancedParams enhancedParams) : Pyin(sampleRate, frameSize, kDefaultMinFreq, kDefaultMaxFreq, enhancedParams) {};;

    Pyin(const Pyin&) = delete;
    Pyin& operator=(const Pyin&) = delete;
    Pyin(Pyin&&) = default;
    Pyin& operator=(const Pyin&&) = delete;
    ~Pyin() = default;

    PyinResult process(const float* audioFrame);

private:
    struct PitchBin {
        std::vector<float> bestFreq;
        std::vector<float> bestFreqProb;
        std::vector<double> binProb;
    };

    const double mSampleRate;
    const int mFrameSize;

    const float mMinFreq;
    const float mMaxFreq;
    const int mMinPeriod;
    const int mMaxPeriod;

    const int mNumBinsHMM;
    std::vector<float> mBuffer;
    std::vector<float> mDiff; // Buffer for difference function
    std::vector<double> mHiddenState[2]; // 0: unvoiced, 1: voiced
    std::vector<double> mNextHiddenState[2]; // 0: unvoiced, 1: voiced
    PitchBin mPitchBins;

    TransitionParams mTransition;
    const double mVoicedTrust;
    const double mYinTrust;

    juce::dsp::FFT mFFT; // for faster auto correlation
};

} // namespace pyin_pitch_detection