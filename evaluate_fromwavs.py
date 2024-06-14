import argparse
import os
import glob
import numpy as np
import kissdsp.io as io

import mir_eval
import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

parser = argparse.ArgumentParser()
parser.add_argument('--wav_dir', default=None, type=str, help='Directory with all the WAV files created by script_test.sh.')
args = parser.parse_args()

if args.wav_dir == None:
  print("A WAV directory is required.")
  exit()

if not os.path.exists(args.wav_dir):
  print("The directory does not exists: "+args.wav_dir)
  exit()

if not os.path.isdir(args.wav_dir):
  print("The provided path is not a directory: "+args.wav_dir)
  exit()

sample_rate = 16000
stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)
pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb')

target_sdr = 0
estima_sdr = 0
target_pesq = 0
target_stoi = 0
estima_pesq = 0
estima_stoi = 0
path_num = 0
for wav_filepath in glob.glob(os.path.join(args.wav_dir,"*.wav")):
  print(wav_filepath)
  [y_target, y_ideal, y_est, y_ref] = io.read(wav_filepath)
  (sdr_t, sir, sar, perm) = mir_eval.separation.bss_eval_sources(y_ideal, y_target)
  (sdr_e, sir, sar, perm) = mir_eval.separation.bss_eval_sources(y_ideal, y_est)

  y_ideal = torch.from_numpy(y_ideal)
  y_target = torch.from_numpy(y_target)
  y_est = torch.from_numpy(y_est)
  
  pesq_t = pesq(y_target,y_ideal).numpy()
  stoi_t = stoi(y_target,y_ideal).numpy()
  pesq_e = pesq(y_est,y_ideal).numpy()
  stoi_e = stoi(y_est,y_ideal).numpy()
  
  print ("   Target SDR : "+("{:.2f}".format( sdr_t[0]))+", Estimation SDR : "+("{:.2f}".format( sdr_e[0]))+", Diff : "+("{:.2f}".format( sdr_e[0]-sdr_t[0])))
  print ("   Target PESQ: "+("{:.2f}".format(pesq_t))+", Estimation PESQ: "+("{:.2f}".format(pesq_e))+", Diff : "+("{:.2f}".format(pesq_e-pesq_t)))
  print ("   Target STOI: "+("{:.2f}".format(stoi_t))+", Estimation STOI: "+("{:.2f}".format(stoi_e))+", Diff : "+("{:.2f}".format(stoi_e-stoi_t)))
  
  target_sdr += sdr_t[0]
  estima_sdr += sdr_e[0]
  target_pesq += pesq_t
  target_stoi += stoi_t
  estima_pesq += pesq_e
  estima_stoi += stoi_e
  path_num += 1

  
target_sdr /= path_num
estima_sdr /= path_num
target_pesq /= path_num
target_stoi /= path_num
estima_pesq /= path_num
estima_stoi /= path_num
print("Average SDR from Target:     "+"{:.2f}".format(target_sdr))
print("Average SDR from Estimation: "+"{:.2f}".format(estima_sdr))
print("Average PESQ from Target:     "+"{:.2f}".format(target_pesq))
print("Average PESQ from Estimation: "+"{:.2f}".format(estima_pesq))
print("Average STOI from Target:     "+"{:.2f}".format(target_stoi))
print("Average STOI from Estimation: "+"{:.2f}".format(estima_stoi))
