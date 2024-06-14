#!/bin/bash

# make sure to add export PYTHONPATH=:${HOME}/MVDRpf/kissdsp if using local installation

#bash script_traineval.sh /home/ar/Data2/databases/mvdrpdf-steernet_db ugru_1-128_1ch
#bash script_test.sh /home/ar/Data2/databases/mvdrpdf-steernet_db ugru_1-128_1ch 2
#python evaluate_fromwavs.py --wav_dir /home/ar/Data2/databases/mvdrpdf-steernet_db/postfiltered/ugru_1-128_1ch | tee ugru_1-128_1ch_results.txt

if [ -z "$1" ]; then
		echo "Need base folder (where the speech text files are) or checkpoint file."
		exit 1
fi

basedir="$1"

dataset_test="$basedir/features/features_test_array.txt"
if [[ ! -f "$dataset_test" ]]; then
		echo "Invalid test features text file: $dataset_test"
		exit 1
fi

num_workers=12 #for testing: 4
batch_size=16
num_epochs=10 #for testing: 9
num_epochs_eval=5 #for testing: 3

#declare -a list_models=("ugru_1-128_1ch")
#declare -a list_models=("ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")

declare -a list_models=("ugru_1-128_1ch" "ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch" "ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch" "ulstm_1-128_1ch" "ulstm_1-128_2ch" "ulstm_1-256_1ch" "ulstm_1-256_2ch" "ulstm_1-512_1ch" "ulstm_1-512_2ch" "ulstm_2-128_1ch" "ulstm_2-128_2ch" "ulstm_2-256_1ch" "ulstm_2-256_2ch" "ulstm_2-512_1ch" "ulstm_2-512_2ch")

for model in "${list_models[@]}"
do

	echo "Doing: $model"
	checkpoints_dir="$basedir/checkpoints/${model}/"
	if [ ! -d "$checkpoints_dir" ]; then
		mkdir -p "$checkpoints_dir"
	fi
	checkpoint_file="$checkpoints_dir/002.bin"

	# training/validation phase
	#if [[ ! -f "$checkpoint_file" ]]; then
	#	python3 ml.py --dataset_train $dataset_train \
	#					--dataset_eval $dataset_valid \
	#					--batch_size $batch_size \
	#					--model $model \
	#					--action traineval \
	#					--num_workers $num_workers \
	#					--num_epochs $num_epochs \
	#					--num_epochs_eval $num_epochs_eval \
	#					--checkpoint_save_dir $checkpoints_dir
	#else
	#	echo "Model already trained."
	#fi

	# evaluation phase
	if [[ ! -f "$checkpoint_file" ]]; then
		echo "Invalid checkpoint bin file: $checkpoint_file"
		echo "	skipping."
		continue
	fi

	results_dir="$basedir/results/${model}_array"
	if [ ! -d "$results_dir" ]; then
		mkdir -p "$results_dir"
	fi
	results_file="$results_dir/improvements.txt"
	csv_file="$results_dir/improvements.csv"

	if [[ ! -f "$csv_file" ]]; then
		python3 ml.py --dataset $dataset_test \
						--model $model \
						--action improvement \
						--checkpoint_load "$checkpoint_file" >> $results_file

		# storing results
		cat $results_file | \
		sed --expression "s/PESQ_start = //g" | \
		sed --expression "s/PESQ_end = //g" | \
		sed --expression "s/STOI_start = //g" | \
		sed --expression "s/STOI_end = //g" | \
		sed --expression "s/SDR_start = //g" | \
		sed --expression "s/SDR_end = //g" | \
		sed --expression "1s/^/pesq_start,pesq_end,stoi_start,stoi_end,sdr_start,sdr_end\n/" > $csv_file
	else
		echo "Model already evaluated."
	fi

done
