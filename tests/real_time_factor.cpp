#include "pyin.h"
#include "utils.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <juce_audio_formats/juce_audio_formats.h>
#include <memory>
#include <tuple>
#include <vector>

juce::File kAudioDatasetDir(AUDIO_DATASET_DIR);
constexpr double kSampleRate = 44100.0;
constexpr int kHopSize = 128; // hop size used in the F0 annotation dataset

int main(void) {
    double totalAudioTime = 0.0f;
    double totalProcessingTime = 0.0f;

    // Find total # of wav files in the dataset
    auto melodyIter = juce::RangedDirectoryIterator(kAudioDatasetDir, false, "*.wav");
    
    std::cout<<"start experiment"<<std::endl;
    for(juce::DirectoryEntry entry : melodyIter)
    {
        // Load melody audio
        juce::File melodyFile = entry.getFile();
        juce::WavAudioFormat wav_format{};
        auto reader = std::unique_ptr<juce::AudioFormatReader>(
            wav_format.createReaderFor(new juce::FileInputStream(melodyFile), true));
        assert(reader->sampleRate == kSampleRate);
        juce::AudioBuffer<float> rawInputAudio(reader->numChannels, reader->lengthInSamples);
        const bool read_result = reader->read(&rawInputAudio, 0, reader->lengthInSamples, 0, true, true);
        assert(read_result);
        totalAudioTime += reader->lengthInSamples / kSampleRate;

        // Convert input audio to mono
        const int frameSize = pyin_pitch_detection::Pyin::getDefaultFrameSize(kSampleRate);
        assert(frameSize/2 >= kHopSize);
        const int numSamples = frameSize + rawInputAudio.getNumSamples(); // first and last frameSize/2 are zero padded 
        juce::AudioBuffer<float> inputAudio(1, numSamples);
        
        inputAudio.clear();
        const float gain = 1.0f / static_cast<float>(rawInputAudio.getNumChannels()); // normalization
        for(int c = 0; c < rawInputAudio.getNumChannels(); ++c)
            inputAudio.addFrom(0, frameSize/2, rawInputAudio, c, 0, rawInputAudio.getNumSamples(), gain);
        
        // Instantiate pyin
        pyin_pitch_detection::Pyin pyin{};
        pyin.prepareToPlay(kSampleRate, frameSize);
        
        // frame-by-frame analysis
        auto start_time = std::chrono::high_resolution_clock::now();
        for(int frameStartIdx = 0; frameStartIdx < inputAudio.getNumSamples() - frameSize; frameStartIdx += kHopSize)
            auto result = pyin.process(inputAudio.getReadPointer(0) + frameStartIdx);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto elapsed = end_time - start_time;
        totalProcessingTime += std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    }

    std::cout<<"finished experiment"<<std::endl;
    std::cout<<"====================================================="<<std::endl;
    std::cout<<"====================================================="<<std::endl;
    std::cout<<"result: "<<std::endl;
    std::cout<<"Total Audio Duration: "<<totalAudioTime<<"s"<<std::endl;
    std::cout<<"Total Processing Time: "<<totalProcessingTime<<"s"<<std::endl;
    const double rtf = totalProcessingTime / totalAudioTime;
    std::cout<<"Real Time Factor: "<<rtf<<std::endl;

}
