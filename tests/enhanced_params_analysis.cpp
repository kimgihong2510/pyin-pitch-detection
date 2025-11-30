#include "pyin.h"
#include "utils.h"

#include <cassert>
#include <iostream>
#include <juce_audio_formats/juce_audio_formats.h>
#include <memory>
#include <tuple>
#include <vector>

juce::File kAudioDatasetDir(AUDIO_DATASET_DIR);
juce::File kLabelDatasetDir(ANNOTATION_DATASET_DIR); // annotation dataset
constexpr double kSampleRate = 44100.0;
constexpr int kHopSize = 128; // hop size used in the F0 annotation dataset

struct FrequencyAccuracy {
    int correctCnt = 0;
    float threshold = 0.0f; // in semitone
};

struct ExperimentResult {
    int voicedToUnvoicedErrorCnt = 0; // voiced segment misclassified to unvoiced 
    int unvoicedToVoicedErrorCnt = 0; // unvoiced segment misclassified to voiced
    int voicedSegmentCnt = 0;
    int unvoicedSegmentCnt = 0;
    float freqErrorSum = 0.0f;
    std::vector<FrequencyAccuracy> freqAccuracyTests{};
};

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cout << "Usage: mdb_melody_synth_analysis <voiced trust> <yin trust> <voiced to unvoiced probability> <unvoiced to voiced probability>" << std::endl;
        return 1;
    }

    const double voicedTrustProb = std::atof(argv[1]);
    const double yinTrustProb = std::atof(argv[2]);
    const double voicedToUnvoicedProb = std::atof(argv[3]);
    const double unvoicedToVoicedProb = std::atof(argv[4]); 

    ExperimentResult experimentRecord{0, 0, 0, 0, 0.0f, {{0, 0.1f}, {0, 0.5f}, {0, 1.0f}}}; // error threshold of 0.1, 0.5, 1.0 semitones

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
    
        // Load f0 label
        auto labelFileName = melodyFile.withFileExtension(".csv").getFileName();
        juce::File labelFile = kLabelDatasetDir.getChildFile(labelFileName);
        assert(labelFile.existsAsFile());
        juce::StringArray labelLines;
        labelFile.readLines(labelLines);
        labelLines.removeEmptyStrings(); // Last String in .csv file is empty

        // Parse and store f0 targets
        std::vector<float> f0Targets;
        for(const juce::String& line: labelLines) {
            juce::StringArray columns;
            columns.addTokens(line, ",", ""); 

            assert(columns.size() == 2);
        
            float f0Target = columns[1].trim().getFloatValue();
            f0Targets.push_back(f0Target);
        }

        // Convert input audio to mono
        const int frameSize = pyin_pitch_detection::Pyin::getDefaultFrameSize(kSampleRate);
        assert(frameSize/2 >= kHopSize);
        const int numSamples = frameSize + (f0Targets.size()-1) * kHopSize; // first and last frameSize/2 are zero padded 
        juce::AudioBuffer<float> inputAudio(1, numSamples);
        inputAudio.clear();

        const int numSamplesToCopy = std::min(rawInputAudio.getNumSamples(), numSamples - frameSize/2);
        const float gain = 1.0f / static_cast<float>(rawInputAudio.getNumChannels()); // normalization
        for(int c = 0; c < rawInputAudio.getNumChannels(); ++c)
            inputAudio.addFrom(0, frameSize/2, rawInputAudio, c, 0, numSamplesToCopy, gain);
        
        // Instantiate pyin
        auto transitionParams = pyin_pitch_detection::TransitionParams{
            1.0f - voicedToUnvoicedProb, 
            unvoicedToVoicedProb,
            voicedToUnvoicedProb,
            1.0f - unvoicedToVoicedProb
        };
        auto enhancedParams = pyin_pitch_detection::EnhancedParams{
            voicedTrustProb,
            yinTrustProb,
            transitionParams
        };
        pyin_pitch_detection::Pyin pyin{};
        pyin.prepareToPlay(kSampleRate, frameSize, 100.0f, 3000.0f, enhancedParams);
        
        // frame-by-frame analysis
        int frameStartIdx = 0;
        for(int targetIdx = 0; targetIdx < f0Targets.size(); ++targetIdx) {
            assert(frameStartIdx + frameSize <= inputAudio.getNumSamples());
            auto result = pyin.process(inputAudio.getReadPointer(0) + frameStartIdx);

            assert(targetIdx < f0Targets.size());
            float f0Target = f0Targets[targetIdx];

            // voiced & unvoiced evaluation
            if(f0Target == 0.0f) { // unvoiced
                ++experimentRecord.unvoicedSegmentCnt;
                if(result.isVoiced)
                    ++experimentRecord.unvoicedToVoicedErrorCnt;
            }
            else if(f0Target != 0.0f) { // voiced
                ++experimentRecord.voicedSegmentCnt;
                if(!result.isVoiced)
                    ++experimentRecord.voicedToUnvoicedErrorCnt;

                // frequency estimation evaluation
                const float errorInSemitone = test_utils::distanceInSemitone(result.freq, f0Target);
                experimentRecord.freqErrorSum += errorInSemitone;
                for(auto& freqTest: experimentRecord.freqAccuracyTests) {
                    const float errorThreshold = freqTest.threshold;
                    freqTest.correctCnt += errorInSemitone < errorThreshold ? 1 : 0;
                }
            }
            frameStartIdx += kHopSize;
        }
    }

    std::cout<<"finished experiment"<<std::endl;
    std::cout<<"====================================================="<<std::endl;
    std::cout<<"====================================================="<<std::endl;
    std::cout<<"result: "<<std::endl;
    std::cout<<"# of labeled frames:\n"<<experimentRecord.voicedSegmentCnt+experimentRecord.unvoicedSegmentCnt<<std::endl;

    // voiced to unvoiced error
    const float VTUError = static_cast<float>(experimentRecord.voicedToUnvoicedErrorCnt) / experimentRecord.voicedSegmentCnt;
    std::cout<<"voiced to unvoiced error:\n"
        <<experimentRecord.voicedToUnvoicedErrorCnt<<" / "
        <<experimentRecord.voicedSegmentCnt<<" = "
        <<VTUError * 100.0f<<"%"<<std::endl; 
    
    // unvoiced to voiced error
    const float UTVError = static_cast<float>(experimentRecord.unvoicedToVoicedErrorCnt) / experimentRecord.unvoicedSegmentCnt;
    std::cout<<"unvoiced to voiced error:\n"
        <<experimentRecord.unvoicedToVoicedErrorCnt<<" / "
        <<experimentRecord.unvoicedSegmentCnt<<" = "
        <<UTVError * 100.0f<<"%"<<std::endl; 
    
    const float meanError = experimentRecord.freqErrorSum / experimentRecord.voicedSegmentCnt;
    std::cout<<"mean of frequency estimation error in semitones:\n"
        <<experimentRecord.freqErrorSum<<" / "<<experimentRecord.voicedSegmentCnt
        <<" = "<<meanError<<" semitones"<<std::endl;

    // frequency estimation accuracy
    for(const auto& test: experimentRecord.freqAccuracyTests) {
        const float accuracy = static_cast<float>(test.correctCnt) / experimentRecord.voicedSegmentCnt;
        std::cout<<"frequency estimation accuracy @ threshold "<<test.threshold<<"semitones:\n"
            <<test.correctCnt<<" / "<<experimentRecord.voicedSegmentCnt
            <<" = "<<accuracy * 100.0f<<"%"<<std::endl;
    }
}
