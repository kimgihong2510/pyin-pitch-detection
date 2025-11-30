#pragma once

#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>

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

    void prepareToPlay(double sampleRate, int frameSize, float minFreq, float maxFreq, EnhancedParams enhancedParams);
    void prepareToPlay(double sampleRate, int frameSize, float minFreq = kDefaultMinFreq, float maxFreq = kDefaultMaxFreq);
    
    PyinResult process(const float* audioFrame);

private:
    struct PitchBin {
        std::vector<float> bestFreq;
        std::vector<float> bestFreqProb;
        std::vector<double> binProb;
    };

    double mSampleRate;
    int mFrameSize;

    float mMinFreq;
    float mMaxFreq;
    int mMinPeriod;
    int mMaxPeriod;

    int mNumBinsHMM;
    std::vector<float> mBuffer;
    std::vector<float> mDiff; // Buffer for difference function
    std::vector<double> mHiddenState[2]; // 0: unvoiced, 1: voiced
    std::vector<double> mNextHiddenState[2]; // 0: unvoiced, 1: voiced
    PitchBin mPitchBins;

    TransitionParams mTransition;
    double mVoicedTrust;
    double mYinTrust;

    std::unique_ptr<juce::dsp::FFT> mFFT;
};

} // namespace pyin_pitch_detection