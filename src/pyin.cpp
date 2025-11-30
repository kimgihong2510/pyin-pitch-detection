#include "pyin.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <juce_dsp/juce_dsp.h>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

namespace pyin_pitch_detection {

namespace {
    static constexpr double kEpsilon = 1e-16;
    static constexpr int kTransitionWidth = 25; // width of pitch transition distribution
    static constexpr float kPitchBinWidth = 0.1f; // 0.1 semitone
    static constexpr float kStepSizeThreshold = 0.01f;
    static constexpr float kPriorGlobalMin = 0.01f; // p_a from EQ. (5) in PYIN paper
    // Precomputed 1.0 - CDF of Beta distribution with mean 0.1 and beta 18
    static constexpr std::array kReversedCDFBeta = {
        1.000000, 0.984434, 0.944401, 0.888188, 0.822152, 0.751094, 0.678570, 0.607148,
        0.538620, 0.474173, 0.414530, 0.360061, 0.310877, 0.266899, 0.227918, 0.193635,
        0.163700, 0.137732, 0.115346, 0.096159, 0.079805, 0.065941, 0.054246, 0.044432,
        0.036235, 0.029422, 0.023786, 0.019146, 0.015343, 0.012240, 0.009721, 0.007685,
        0.006048, 0.004736, 0.003692, 0.002864, 0.002210, 0.001697, 0.001296, 0.000985,
        0.000744, 0.000559, 0.000417, 0.000310, 0.000229, 0.000168, 0.000122, 0.000088,
        0.000064, 0.000045, 0.000032, 0.000023, 0.000016, 0.000011, 0.000007, 0.000005,
        0.000003, 0.000002, 0.000001, 0.000001, 0.000001
    };

    // datas are interleaved complex numbers (real, imag, real, ...)
    static inline void conjugate(float* data, int length) {
        assert(length % 2 == 0); // data is interleaved
        for (int i = 1; i < length; i+=2)
            data[i] = -data[i];
    }

    // datas are interleaved complex numbers (real, imag, real, ...)
    static inline void multiply(float* a, const float* b, int length) {
        assert(length % 2 == 0); // data is interleaved 
        for (int i = 0; i < length; i+=2) {
            const float tmp = a[i]*b[i] - a[i+1]*b[i+1];
            a[i+1] = a[i]*b[i+1] + a[i+1]*b[i];
            a[i] = tmp;
        }
    }

    constexpr int toIdx(Pyin::VoiceState s) {
        return static_cast<int>(s);
    }

    static inline float parabolicInterpolation(const std::vector<float>& diff, int tau) {
        assert(0 <= tau && tau < diff.size());
        if (tau == 0 || tau == diff.size() - 1)
            return static_cast<float>(tau);

        const float s0 = diff[tau - 1];
        const float s1 = diff[tau];
        const float s2 = diff[tau + 1];
        
        const float denominator = (s0 - 2.0f * s1 + s2);

        return denominator != 0.0f ? tau - 0.5f * (s2 - s0) / denominator : static_cast<float>(tau);
    }
    
    static float distanceInSemitone(float freq1, float freq2) {
        assert(freq1 > 0.0f && freq2 > 0.0f);
        return std::abs(std::log2(freq1/freq2)) * 12.0f;
    }

    static int findNearestBin(float freq, float minFreq, int numBins) {
        assert(minFreq <= freq);
        const float dist = distanceInSemitone(freq, minFreq);

        int binIdx = static_cast<int>(std::round(dist / kPitchBinWidth));  

        assert(0 <= binIdx && binIdx < numBins);
        return binIdx;
    }

    template<typename T>
    static T clamp(T n, T min, T max) {
        return std::min(std::max(n, min), max);
    }
} // namespace

void Pyin::prepareToPlay(double sampleRate, int frameSize, float minFreq, float maxFreq, EnhancedParams enhancedParams) {
    assert(sampleRate/2.0 > maxFreq && maxFreq > minFreq && minFreq > 0.0f); // maximum frequency < nyquist frequency
    assert(frameSize > 2 * mMaxPeriod); 
    assert(frameSize % 2 == 0);

    mSampleRate = sampleRate;
    mFrameSize = frameSize;
    mMinFreq = minFreq;
    mMaxFreq = maxFreq;
    mMinPeriod = static_cast<int>(std::ceil(sampleRate / maxFreq)); 
    mMaxPeriod = static_cast<int>(std::floor(sampleRate / minFreq));
    mNumBinsHMM = static_cast<int>(std::ceil(distanceInSemitone(minFreq, maxFreq)/kPitchBinWidth)+1);
    
    mBuffer.resize(frameSize * 2);
    mDiff.resize(frameSize * 2);

    mHiddenState[0].resize(mNumBinsHMM, 0.5 / mNumBinsHMM);
    mHiddenState[1].resize(mNumBinsHMM, 0.5 / mNumBinsHMM); 

    mNextHiddenState[0].resize(mNumBinsHMM);
    mNextHiddenState[1].resize(mNumBinsHMM);

    mPitchBins.bestFreq.resize(mNumBinsHMM);
    mPitchBins.bestFreqProb.resize(mNumBinsHMM);
    mPitchBins.binProb.resize(mNumBinsHMM);

    mTransition = enhancedParams.transitionParams;
    mVoicedTrust = enhancedParams.voicedTrust;
    mYinTrust = enhancedParams.yinTrust;
    mFFT = std::make_unique<juce::dsp::FFT>(static_cast<int>(std::log2(frameSize)));

    double probSum = mTransition.voicedToUnvoiced + mTransition.voicedToVoiced;
    const double eps = 1e-8;
    assert(std::abs(probSum - 1.0f) < eps); // sum of probability must equal to 1
    probSum = mTransition.unvoicedToUnvoiced + mTransition.unvoicedToVoiced;
    assert(std::abs(probSum - 1.0f) < eps); // sum of probability must equal to 1

    assert(0.0 < mYinTrust && mYinTrust < 1.0);
    assert(0.0 < mVoicedTrust && mVoicedTrust < 1.0);

    // assign initial best frequencies to each pitch bins
    double bestFreq = minFreq;
    for(int i=0; i<mNumBinsHMM; i++) {
        mPitchBins.bestFreq[i] = minFreq;
        minFreq *= exp2(1.0f/12.0f * kPitchBinWidth);
    }
}

void Pyin::prepareToPlay(double sampleRate, int frameSize, float minFreq, float maxFreq) {
    prepareToPlay(sampleRate, frameSize, minFreq, maxFreq, EnhancedParams{});
}

int Pyin::getSmallestFrameSize(double sampleRate, float minFreq){
    assert(minFreq > 0.0f);
    int maxPeriod = static_cast<int>(std::floor(sampleRate / minFreq));

    // find next power of two
    int result = 1;
    for(; result <= 2*maxPeriod; result*=2) {};
    return result;
}

int Pyin::getDefaultFrameSize(double sampleRate, float minFreq){
    return getSmallestFrameSize(sampleRate, minFreq) * 2;
}

EnhancedParams Pyin::getDefaultEnhancedParams() {
    return EnhancedParams{};
}

PyinResult Pyin::process(const float* audioFrame) {
    // 1. DIFFERENCE FUNCTION (YIN)

    // FFT based auto-correlation to calculate the difference function
    // Audio frame and its half(mBuffer) are cross-correlated.
    std::copy(audioFrame, audioFrame + mFrameSize, mDiff.begin());
    std::fill(mDiff.begin() + mFrameSize, mDiff.end(), 0.0f);

    assert(mFrameSize % 2 == 0);
    const int kernelSize = mFrameSize / 2;
    assert(kernelSize >= mMaxPeriod); 
    std::copy(audioFrame, audioFrame + kernelSize, mBuffer.begin());
    std::fill(mBuffer.begin() + kernelSize, mBuffer.end(), 0.0f);
    
    mFFT->performRealOnlyForwardTransform(mDiff.data());
    mFFT->performRealOnlyForwardTransform(mBuffer.data());
    
    conjugate(mDiff.data(), mFrameSize*2);
    multiply(mDiff.data(), mBuffer.data(), mFrameSize*2);
    
    mFFT->performRealOnlyInverseTransform(mDiff.data());
    
    // find the power terms and calculate difference function 
    // This corresponds to EQ. (7) from YIN paper
    const float energyAtZero = mDiff[0];
    float energyAtTau = mDiff[0];
    for (int tau = 0; tau <= mMaxPeriod; ++tau) {
        mDiff[tau] = energyAtZero + energyAtTau - mDiff[tau] * 2.0f;
        mDiff[tau] = std::max(mDiff[tau], 0.0f); // It may be < 0 due to numerical error
        energyAtTau += (audioFrame[tau + kernelSize] * audioFrame[tau + kernelSize] - audioFrame[tau] * audioFrame[tau]);
    }

    // 2. CUMULATIVE MEAN NORMALIZED DIFFERENCE FUNCTION (YIN) 
    //    and PROBABILISTIC THRESHOLD (PYIN)
    std::fill(mPitchBins.bestFreqProb.begin(), mPitchBins.bestFreqProb.end(), 0.0f);
    std::fill(mPitchBins.binProb.begin(), mPitchBins.binProb.end(), 0.0f);

    float curNormalizedDiff = 0.0f;
    float nextNormalizedDiff = 1.0f;
    float globalMinNormalizedDiff = std::numeric_limits<float>::max();
    int globalMinPitchBinIdx = -1;
    float runningSumOfDiff = 0.0f;
    double lastRevCDF = 0.0f;

    for(int tau = 0; tau < mMaxPeriod; ++tau) {
        const bool prevIsDecreasing = curNormalizedDiff >= nextNormalizedDiff ? true : false;
        curNormalizedDiff = nextNormalizedDiff;
        runningSumOfDiff += mDiff[tau + 1];
        nextNormalizedDiff = mDiff[tau + 1] / ((runningSumOfDiff / (tau + 1))); 
        
        // Check if 
        // 1) d'(tau) is a local minimum 
        // 2) d'(tau) is the smallest among all local minima before tau
        if(tau >= mMinPeriod && (prevIsDecreasing && nextNormalizedDiff >= curNormalizedDiff) && globalMinNormalizedDiff > curNormalizedDiff) {
            const int thresholdIdx = static_cast<int>(std::floor(curNormalizedDiff / kStepSizeThreshold));
            constexpr int maxIdx = kReversedCDFBeta.size(); // after this index, all values are zeros anyway
            const double revCDF = thresholdIdx < maxIdx ? kReversedCDFBeta[thresholdIdx] : 0.0f;
            const double prob = revCDF - lastRevCDF;

            const float interpolatedTau = parabolicInterpolation(mDiff, tau);
            const float freq = clamp(static_cast<float>(mSampleRate) / interpolatedTau, mMinFreq, mMaxFreq); 
            const int pitchBinIdx = findNearestBin(freq, mMinFreq, mNumBinsHMM);
            // Use ">=" instead of ">" because equal probabilities(mostly zero) can occur due to numerical precision limits. 
            // The later freq should win since its theoretical probability is higher.
            if(prob >= mPitchBins.bestFreqProb[pitchBinIdx]) { 
                mPitchBins.bestFreq[pitchBinIdx] = freq;
                mPitchBins.bestFreqProb[pitchBinIdx] = prob;
            }
            mPitchBins.binProb[pitchBinIdx] += prob;
            lastRevCDF = revCDF;
            
            globalMinPitchBinIdx = pitchBinIdx;
            globalMinNormalizedDiff = curNormalizedDiff; 
        }
    }

    assert(lastRevCDF <= 1.0f);
    double globalMinProb = 0.0f;
    if(globalMinPitchBinIdx >= 0) {
        globalMinProb = (1.0f - lastRevCDF) * kPriorGlobalMin;
        mPitchBins.binProb[globalMinPitchBinIdx] += globalMinProb;
    }
    
    // calculate observation probability for unvoiced state
    assert(lastRevCDF + globalMinProb <= 1.0f);
    const double sumVoicedObservationProb = globalMinProb + lastRevCDF;
    const double observationUnvoicedProb = (1.0f - sumVoicedObservationProb) / mNumBinsHMM * mYinTrust + (1.0 - mVoicedTrust) * (1.0 - mYinTrust); 
    
    // 3. HMM(Viterbi)  
    PyinResult result = {0.0f, false};
    double maxPathProb = -1.0f;
    double sumProbHMM = 0.0f;
    for(int binIdx = 0; binIdx < mNumBinsHMM; ++binIdx) {
        // Viterbi update for the pitch bin
        assert(kTransitionWidth % 2 == 1);
        constexpr int halfWidth = (kTransitionWidth+1) / 2;
        constexpr double sqrHalfWidth = static_cast<double>(halfWidth * halfWidth);

        double maxPrevVoicedProb = kEpsilon;
        double maxPrevUnvoicedProb = kEpsilon;

        // Search through previous state probabilities within pitch transition range
        const int jMin = -std::min(halfWidth-1, binIdx);
        const int jMax = std::min(halfWidth, mNumBinsHMM-binIdx); 
        for(int j = jMin; j < jMax; ++j) {
            const int prevStateBinIdx = binIdx + j;
            assert(0 <= prevStateBinIdx && prevStateBinIdx < mNumBinsHMM);
            
            const double pitchTransitionProb = (halfWidth - std::abs(j)) / sqrHalfWidth;
            maxPrevVoicedProb = std::max(maxPrevVoicedProb, 
                mHiddenState[toIdx(VoiceState::VOICED)][prevStateBinIdx] * pitchTransitionProb);
            maxPrevUnvoicedProb = std::max(maxPrevUnvoicedProb,
                mHiddenState[toIdx(VoiceState::UNVOICED)][prevStateBinIdx] * pitchTransitionProb); 
        }

        // next voiced state
        const double observationVoicedProb = mPitchBins.binProb[binIdx] * mYinTrust + mVoicedTrust * (1.0 - mYinTrust);
        double pathProb = std::max(
            maxPrevVoicedProb * mTransition.voicedToVoiced * observationVoicedProb,
            maxPrevUnvoicedProb * mTransition.unvoicedToVoiced * observationVoicedProb
        );
        pathProb = clamp(pathProb, kEpsilon, 1.0); 
        mNextHiddenState[toIdx(VoiceState::VOICED)][binIdx] = pathProb; 
        if(pathProb > maxPathProb) {
            result = {mPitchBins.bestFreq[binIdx], true};
            maxPathProb = pathProb;
        }
        sumProbHMM += pathProb;
         
        // next unvoiced state
        pathProb = std::max(
            maxPrevUnvoicedProb * mTransition.unvoicedToUnvoiced * observationUnvoicedProb,
            maxPrevVoicedProb * mTransition.voicedToUnvoiced * observationUnvoicedProb
        );
        pathProb = clamp(pathProb, kEpsilon, 1.0); 
        mNextHiddenState[toIdx(VoiceState::UNVOICED)][binIdx] = pathProb;
        if(pathProb > maxPathProb) {
            result = {mPitchBins.bestFreq[binIdx], false};
            maxPathProb = pathProb;
        }
        sumProbHMM += pathProb;
    }
    // normalize HMM bins
    for(const auto state: {VoiceState::UNVOICED, VoiceState::VOICED}) 
        for(int binIdx = 0; binIdx < mNumBinsHMM; ++binIdx)
            mNextHiddenState[toIdx(state)][binIdx] /= sumProbHMM;

    std::swap(mNextHiddenState, mHiddenState);

    // frequency candidates has been clamped in step 2
    assert(mMinFreq <= result.freq && result.freq <= mMaxFreq);

    return result;
}

} // namespace pyin_pitch_detection