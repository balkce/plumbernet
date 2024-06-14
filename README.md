# PlumberNet

Leakage removal from speech enhancement.

## Package requirements

The following packages are required:

```
torch
pesq
torchmetrics[audio]
matplotlib
numpy
kissdsp
progressbar
progressbar2
mir_eval
librosa
```

[comment]: <> (pypesq)
[comment]: <> (pystoi)

## Preparing SteerNet weights and dataset

First, it is required to clone the SteerNet github repository locally in `${steernet_dir}`:
```
cd <parent directory to ${steernet_dir}>
git clone https://github.com/FrancoisGrondin/steernet.git
```

The code that creates the SteerNet-based features imports the SteerNet python library automatically, depending of where it is located. To facilitate this, once the SteerNet github repository is cloned, create a symbolic link named `steernet` to point to the `python` folder inside it:
```
cd ${steernet_dir}
ln -s python ./steernet
```

Then, carry out the data creation process as established in the [SteerNet repository](https://github.com/FrancoisGrondin/steernet), using the included scripts in that repository. In summary:
1. Create the room impulse responses (RIRs) as explained in the README file located in `${steernet_dir}/octave/rir` directory. Basically, to create 1000 RIRs:
```
cd ${steernet_dir}/octave/rir
octave init.m ${rir_directory}
mkoctfile --mex rir_generator.cpp 
octave simulate.m ${rir_directory} pair 1000
```

1. Download the librispeech corpus to `${root_speech}`.
1. Prepare data for training as explained in the README file located in `${steernet_dir}/python` directory. Basically, to create 100000 recordings:
```
cd ${steernet_dir}/python
python3 plan_speech.py --root ${root_speech_dir} --json json/speech.json > ${speech_meta_txt}
python3 plan_farfield.py --root ${rir_directory} --json json/farfield.json > ${farfield_meta_txt}
python3 plan_audio.py --speech ${speech_meta_txt} --farfield ${farfield_meta_txt} --count 100000 > ${audio_meta_txt}
```

You do not need to re-train the SteerNet; we will use the one located in `${steernet_dir}/trained/blstm_epoch020.bin`.


## Preparing base directory structure and generating features

The base directory in which all of the files are stored (training data, feature text files, checkpoints, results, etc.) should have the following structure:

```
${base_dir}/
  features/
    features_test.txt
    features_train.txt
    features_valid.txt
  data/
    test/
      *.wav
    train/
      *.wav
    valid/
      *.wav
  checkpoints/
    model1/
      *.bin
    model2/
      *.bin
    ...
  results/
    model1/
      *.wav
    model2/
      *.wav
    ...
```

To create this structure, run the `script_basedir` script that is on `${plumbernet_dir}`, the root directory of this cloned repository:
```
cd ${plumbernet_dir}
bash script_basedir.sh ${base_dir}
```

Once the SteerNet dataset is created and an appropriate directory structure is under `${base_dir}`, you could create the features to train PlumberNet running the following command:

```
cd ${plumbernet_dir}
python3 features.py --steernet_basedir ${steernet_dir} --index_start 1 --index_stop 1000 --output ${base_dir}/data/train
```

This will create the features using the first `1000` recordings from the SteerNet dataset. This is convenient as multiple threads can be started to speed up features generation. For instance, each one of the following lines can be executed each in a different terminal:

```
python3 features.py --steernet_basedir ${steernet_dir} --index_start 1    --index_stop 1000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 1001 --index_stop 2000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 2001 --index_stop 3000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 3001 --index_stop 4000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 4001 --index_stop 5000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 5001 --index_stop 6000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 6001 --index_stop 7000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 7001 --index_stop 8000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 8001 --index_stop 9000  --output ${base_dir}/data/train
python3 features.py --steernet_basedir ${steernet_dir} --index_start 9001 --index_stop 10000 --output ${base_dir}/data/train
```

The same idea can be applied to the validation and testing features.

To create 1000 validation features, run:

```
python3 features.py --steernet_basedir ${steernet_dir} --index_start 1 --index_stop 1000 --output ${base_dir}/data/valid
```

To create 1000 testing features, run:

```
python3 features.py --steernet_basedir ${steernet_dir} --index_start 1 --index_stop 1000 --output ${base_dir}/data/test
```

Once this is done, index the features for all files and save them in text files that will be used to load the content during training, validating and testing:

```
find ${base_dir}/data/train -name "*.wav" > ${base_dir}/features/features_train.txt
find ${base_dir}/data/valid -name "*.wav" > ${base_dir}/features/features_valid.txt
find ${base_dir}/data/test  -name "*.wav" > ${base_dir}/features/features_test.txt
```

## Training

The model can be trained as follows (for instance with batch size of 16, using 16 workers/cores, for 10 epochs):

```
python3 ml.py --dataset ${base_dir}/features/features_train.txt --batch_size 16 --action train --num_workers 16 --num_epochs 10 --checkpoint_save <checkpoint_bin_file>
```

The trained parameters are saved in a binary file denoted by `<checkpoint_bin_file>`.

## Validation

It is possible to measure the loss of a specific model based on the saved checkpoint and using the validation set:

```
python3 ml.py --dataset ${base_dir}/features/features_valid.txt --batch_size 16 --action eval --num_workers 16 --checkpoint_load <checkpoint_bin_file>
```

## Testing

The model can be used to generate masks and save the results to png figures on the disk in the directory `<output_png_directory>`. Producing time-domain waveforms with and without the estimated mask will be done soon.

```
python3 ml.py --dataset ${base_dir}/features/features_test.txt --action test --checkpoint_load <checkpoint_bin_file> --output_dir <output_png_directory>
```

## Using the included scripts

To create the features files, given that the SteerNet dataset has been created, as well as the `${base_dir}` directory structure, run:
```
bash script_features.sh ${base_dir}
```

You are welcome to modify the following variables in `script_features.sh`: `trainnum` is the number of data points for training, `validnum` is the number of data points for evaluation/validation, and `testnum` is the number of data points for testing. These number should appropriately chosen given the amount of recordings were created when running the `plan_audio` file from SteerNet.

To train a model from scratch, with an evaluation at every given number of epochs, run:
```
bash script_traineval.sh ${base_dir} ${modelname}
```

Where `${modelname}` is the name of the model to train. Valid choices are:
```
ugru_1-128_1ch, ugru_1-128_2ch, ugru_1-256_1ch, ugru_1-256_2ch, ugru_1-512_1ch, ugru_1-512_2ch,
ugru_2-128_1ch, ugru_2-128_2ch, ugru_2-256_1ch, ugru_2-256_2ch, ugru_2-512_1ch, ugru_2-512_2ch,
ulstm_1-128_1ch, ulstm_1-128_2ch, ulstm_1-256_1ch, ulstm_1-256_2ch, ulstm_1-512_1ch, ulstm_1-512_2ch,
ulstm_2-128_1ch, ulstm_2-128_2ch, ulstm_2-256_1ch, ulstm_2-256_2ch, ulstm_2-512_1ch, ulstm_2-512_2ch
```

You are welcome to modify the following variables in `script_traineval.sh`: `num_workers` is the number of threads to use while training and evaluating, `batch_size` is the size of data points per batch, `num_epochs` is the total number of epochs to run (an epoch being running through all of the data points in `features_train.txt`), `num_epochs_eval` is the number of epochs to run after which an evaluation is carried out and a checkpoint is saved.

To train a model from a given checkpoint, with an evaluation at every given number of epochs, run:
```
bash script_traineval.sh ${path_to_checkpoint}
```

The path to the base directory is assumed from the checkpoint path, assuming the base directory structure is respected.


## To evaluate trained models with various array geometries

The SteerNet dataset was created and trained for 2-microphone array. The RIR created with the `simulated.m` octave script use the `pair` argument to do so.

However, it is also of interest to evaluate the trained PlumberNet with various other type of geometries. To do this, you should create another testing dataset for this purpose. Fortunately, SteerNet already provides tools to do this, but requires to run its dataset instructions with other types of geometries. To facilitate this, the following two scripts can be used.

To create 100 RIRs for each geometry type (other than `pair`) in another directory `${rir_array_directory}`, run:

```
bash script_array_create_steernet_01-rirs.sh ${rir_array_directory} ${steernet_dir}/octave/rir/ 100
```

Assuming you have already run the `plan_speech.py` step to create the SteerNet database, you can create the farfield meta file `${steernet_dir}/array_farfield_meta.txt` and the audio meta file `${steernet_dir}/array_audio_meta.txt` for these other array geometries by running:

```
bash script_array_create_steernet_02-recordings.sh ${rir_array_directory} ${steernet_dir}/python/ ${root_speech_dir} 100
```

Since there are 6 geometries in SteerNet, and we created 100 RIRs per geometries, and if we want 2 recordings per RIR, to create 1200 array features, run:

```
python3 features.py --steernet_basedir ${steernet_dir} --audio_meta ${steernet_dir}/array_audio_meta.txt --index_start 1 --index_stop 1200 --output ${base_dir}/data_test_array
```

Once finished, create the test array features text file by running:

```
find ${base_dir}/data_test_array  -name "*.wav" > ${base_dir}/features/features_test_array.txt
```

And to evaluate a given model `${model}` at a given checkpoint number `${checkpoint_number}`, run:

```
bash script_test_array.sh ${base_dir} ${model} ${base_dir}/features/features_test_array.txt ${checkpoint_number}
python3 evaluate_fromwavs.py --wav_dir ${base_dir}/postfiltered/${model}_array
```

This will create audio files and spectrograms, as well as provide various speech enhancement metrics such as SDR, PESQ and STOI.

Or, you could run the following script that will evaluate all the trained models using the created array test features:

```
bash script_array_eval_everything_array.sh ${base_dir}
```

