import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle
import soundfile as sf

import preprocess
from trainingDataset import trainingDataset
from model_tf import Generator, Discriminator
from tqdm import tqdm

def preprocessing(dir, cache_folder):
    num_mcep = 36
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    print("Starting to prepocess data.......")
    start_time = time.time()

    wavs = preprocess.load_wavs(wav_dir=dir, sr=sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
        wave=wavs, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)

    print("Log Pitch A")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))

    coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A,
             std_A=log_f0s_std_A)

    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_A=coded_sps_A_mean,
             std_A=coded_sps_A_std)

    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(cache_folder, "coded_sps_A_norm.pickle"))

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)

class CycleGANConversion(object):
    def __init__(self,
                 logf0s_normalization,
                 mcep_normalization,
                 coded_sps_A_norm,
                 model_checkpoint,
                 validation_A_dir,
                 output_A_dir,
                 model_to_use,
                 direction):
        self.dataset_A = self.loadPickleFile(coded_sps_A_norm)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.direction = direction
        # Speech Parameters
        logf0s_normalization = np.load(logf0s_normalization)
        if self.direction:
            self.log_f0s_mean_A = logf0s_normalization['mean_A']
            self.log_f0s_std_A = logf0s_normalization['std_A']
            self.log_f0s_mean_B = logf0s_normalization['mean_B']
            self.log_f0s_std_B = logf0s_normalization['std_B']
            mcep_normalization = np.load(mcep_normalization)
            self.coded_sps_A_mean = mcep_normalization['mean_A']
            self.coded_sps_A_std = mcep_normalization['std_A']
            self.coded_sps_B_mean = mcep_normalization['mean_B']
            self.coded_sps_B_std = mcep_normalization['std_B']
        else:
            self.log_f0s_mean_A = logf0s_normalization['mean_B']
            self.log_f0s_std_A = logf0s_normalization['std_B']
            self.log_f0s_mean_B = logf0s_normalization['mean_A']
            self.log_f0s_std_B = logf0s_normalization['std_A']
            mcep_normalization = np.load(mcep_normalization)
            self.coded_sps_A_mean = mcep_normalization['mean_B']
            self.coded_sps_A_std = mcep_normalization['std_B']
            self.coded_sps_B_mean = mcep_normalization['mean_A']
            self.coded_sps_B_std = mcep_normalization['std_A']

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)

        # To Load save previously saved models
        self.modelCheckpoint = model_checkpoint
        os.makedirs(self.modelCheckpoint, exist_ok=True)

        # Validation set Parameters
        self.validation_A_dir = validation_A_dir
        self.output_A_dir = output_A_dir
        os.makedirs(self.output_A_dir, exist_ok=True)

        if model_to_use is not None:
            self.loadModel(model_to_use)


    def convert(self):
        num_mcep = 36
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Validation Data B from A...")
        for file in os.listdir(validation_A_dir):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            print(self.log_f0s_mean_A)
            print(self.log_f0s_std_A)
            print(self.log_f0s_mean_B)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)

            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            sf.write(file=os.path.join(output_A_dir, os.path.basename(file)),
                                     data=wav_transformed,
                                     samplerate=sampling_rate)

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        if self.direction:
            self.generator_A2B.load_state_dict(
                state_dict=checkPoint['model_genA2B_state_dict'])
        else:
            self.generator_A2B.load_state_dict(
                state_dict=checkPoint['model_genB2A_state_dict'])


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(
    #    description='Prepare data')
    #train_A_dir_default = '/tmp/pycharm_project_279/data/shiri/'
    #cache_folder_default = '/tmp/pycharm_project_279/cache/'

    #parser.add_argument('--dir', type=str,
    #                    help="Directory for source voice sample", default=train_A_dir_default)
    #parser.add_argument('--cache_folder', type=str,
    #                    help="Store preprocessed data in cache folders", default=cache_folder_default)
    #argv = parser.parse_args()

    #dir = argv.dir
    #cache_folder = argv.cache_folder
    #
    #preprocessing(dir, cache_folder)
    #
    parser = argparse.ArgumentParser(
        description="convert sound")

    # direction of the conversion 1= a to b, 0= b to a:
    direction = 0

    place = '/tmp/pycharm_project_279'
    logf0s_normalization_default = place + '/cache_interspeech/logf0s_normalization.npz'
    mcep_normalization_default = place + '/cache_interspeech/mcep_normalization.npz'
    model_checkpoint = place + '/model_checkpoint/'
    model_to_use = place + '/model_checkpoint/_CycleGAN_CheckPoint_interspeech'

    validation_A_dir_default = place + '/data/mix/'
    output_A_dir_default = place + '/converted_sound/mix_to_interview'

    if direction:
        coded_sps_A_norm = place + '/cache_interspeech/coded_sps_A_norm.pickle'
    else:
        coded_sps_A_norm = place + '/cache_interspeech/coded_sps_B_norm.pickle'

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--model_to_use', type=str,
                        help="model to use",
                        default=model_to_use)
    parser.add_argument('--validation_A_dir', type=str,
                        help="validation set for sound source A", default=validation_A_dir_default)
    parser.add_argument('--output_A_dir', type=str,
                        help="output for converted Sound Source A", default=output_A_dir_default)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    coded_sps_A_norm = argv.coded_sps_A_norm
    model_checkpoint = argv.model_checkpoint
    model_to_use = argv.model_to_use

    validation_A_dir = argv.validation_A_dir
    output_A_dir = argv.output_A_dir


    # Check whether following cached files exists
    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")


    cycleGANconv = CycleGANConversion(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_A_norm=coded_sps_A_norm,
                                model_checkpoint=model_checkpoint,
                                validation_A_dir=validation_A_dir,
                                output_A_dir=output_A_dir,
                                model_to_use=model_to_use,
                                direction=direction)
    cycleGANconv.convert()
