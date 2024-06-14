#!/bin/bash

# make sure to have run first:
#   bash create_array_steernet_01-rirs.sh

if [ -z "$1" ]; then
    echo "Need RIR folder created by create_array_steernet_01-rirs."
    exit 1
fi

if [[ -d "$1" ]]; then
    rir_dir="$1"
else
    echo "RIR folder not found."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Need steernet python directory."
    exit 1
fi

if [[ -d "$2" ]]; then
    steernet_dir="$2"
else
    echo "steernet python directory not found."
    exit 1
fi

if [[ -d "$3" ]]; then
    librispeech_dir="$3"
else
    echo "librispeech directory not found."
    exit 1
fi

if [ -z "$4" ]; then
    echo "Need number of recordings per type of array."
    exit 1
fi

recording_num="$4"

#assuming that speech meta file has already been created to train plumbernet

echo "Running plan_farfield.py..."
python3 "$steernet_dir/plan_farfield.py" --root "$rir_dir" --json "$steernet_dir/json/farfield.json" > "$steernet_dir/array_farfield_meta.txt"

echo "Running plan_audio.py..."
python3 "$steernet_dir/plan_audio.py" --speech "$steernet_dir/speech_meta.txt" --farfield "$steernet_dir/array_farfield_meta.txt" --count "$recording_num" > "$steernet_dir/array_audio_meta.txt"

echo "To create array features, run:"
echo "python3 features_steernet.py --steernet_basedir \"$steernet_dir\" --audio_meta array_audio_meta.txt --index_start 1 --index_stop $4 --output data_test_array-dir"

echo "To create the features text file:"
echo "find data_test_array -name \"*.wav\" > features_test_array.txt"

echo "To evaluate, run:"
echo "bash script_test_array.sh base_dir ugru_2-256_2ch features_test_array.txt 2"
echo "python evaluate_fromwavs.py --wav_dir base_dir/postfiltered/ugru_2-256_2ch_array | tee ugru_2-256_2ch_array_results.txt"
