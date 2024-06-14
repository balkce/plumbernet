#!/bin/bash

# make sure to have run first:
#   cd "$steernet_rir_dir"
#   octave init.m "$rir_dir"
#   mkoctfile --mex rir_generator.cpp 

if [ -z "$1" ]; then
    echo "Need RIR output folder."
    exit 1
fi

if [[ -d "$1" ]]; then
    rir_dir="$1"
else
    echo "RIR output folder not found."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Need steernet rir directory."
    exit 1
fi

if [[ -d "$2" ]]; then
    steernet_rir_dir="$2"
else
    echo "steernet rir directory not found."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Need number of recordings per type of array."
    exit 1
fi

recording_num="$3"

#mic_arrays=("pair" "respeaker_usb" "respeaker_core" "matrix_creator" "matrix_voice" "minidsp_uma" "microsoft_kinect")
mic_arrays=("respeaker_usb" "respeaker_core" "matrix_creator" "matrix_voice" "minidsp_uma" "microsoft_kinect")
for mic_array in ${mic_arrays[@]}; do
    echo $mic_array
    (cd "$steernet_rir_dir"; octave simulate.m "$rir_dir" "$mic_array" "$recording_num") &
done







