import librosa  # for audio processing
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import os
import math

import preprocess
import scipy
import pandas as pd


def spectrogram(wave):
    S = librosa.stft(wave)
    fig, ax = plt.subplots()
    img = display.specshow(librosa.amplitude_to_db(S,
                                                    ref=np.max),
                            y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

def plot_spectrogram(wave):
    fig, ax = plt.subplots()
    img = display.specshow(librosa.amplitude_to_db(wave,
                                                    ref=np.max),
                            y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

def avg_spectrogram(audio_path, label):
    labels = os.listdir(audio_path)
    for i in range(len(labels)):
        samples, sample_rate = librosa.load(audio_path + f'/{label}/{label}_{i + 1}.wav', sr=8000)
        samples = np.resize(samples, 24000)
        spec = librosa.stft(samples)
        if i == 0:
            avg_spec = spec
        else:
            avg_spec = np.array(avg_spec) + np.array(spec)
    avg_spec = avg_spec / len(labels)
    #return avg_spec
    plot_spectrogram(avg_spec)

def decompose(filePath):
    sampling_rate = 16000
    frame_period = 5.0
    wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
    wav = preprocess.wav_padding(wav=wav,
                                 sr=sampling_rate,
                                 frame_period=frame_period,
                                 multiple=4)
    f0, timeaxis, sp, ap = preprocess.world_decompose(
        wav=wav, fs=sampling_rate, frame_period=frame_period)
    return f0, timeaxis, sp, ap

def plot_path(D, wp, Y):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                   ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='Source', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]],
              title='Matching cost function')
    fig.show()

def compare_two():
    audio_path = '/tmp/pycharm_project_279/data'
    samples, sample_rate = librosa.load(audio_path + '/shiri/interview_bibi_1.wav', sr=8000)
    spec1 = librosa.stft(samples)
    print(spec1.shape)
    #plot_spectrogram(spec1)
    #f0, timeaxis, sp1, ap = decompose(audio_path + f'/shiri/interview_bibi_1.wav')
    samples2, sample_rate = librosa.load('/tmp/pycharm_project_279/converted_sound/interview/interview_bibi_1.wav', sr=8000)
    samples2 = np.resize(samples2, samples.shape)
    spec2 = librosa.stft(samples2)
    print(spec2.shape)
    #f0, timeaxis, sp2, ap = decompose('/tmp/pycharm_project_279/converted_sound/interview/interview_bibi_1.wav')
    D, wp = librosa.sequence.dtw(spec1, spec2, subseq=False)
    plot_path(D, wp, spec2)
    #plot_spectrogram(spec2)

    #plot_spectrogram(np.abs(spec1-spec2))

def get_f0s(logf0s_normalization):
    logf0s_normalization = np.load(logf0s_normalization)
    log_f0s_mean_A = logf0s_normalization['mean_A']
    log_f0s_std_A = logf0s_normalization['std_A']
    log_f0s_mean_B = logf0s_normalization['mean_B']
    log_f0s_std_B = logf0s_normalization['std_B']
    return log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B

def normalize(fft):
    std = np.std(fft)
    mean = np.mean(np.absolute(fft))
    signs = np.real(np.sign(fft))
    return ((np.absolute(fft) - mean)/std) * signs

def get_mean(fft, bottom, top):
    x, fft = get_x(fft)
    i_bottom = np.where(x == bottom)
    i_bottom = int(i_bottom[0])
    i_top = np.where(x == top)
    i_top = int(i_top[0])
    norm_fft = normalize(fft)
    mean = np.mean(np.absolute(norm_fft[i_bottom:i_top]))
    return mean

def get_mean_ratio(fft, bottom1, top1, bottom2, top2):
    x, fft = get_x(fft)
    i_bottom1 = np.where(x == bottom1)
    i_bottom1 = int(i_bottom1[0])
    i_top1 = np.where(x == top1)
    i_top1 = int(i_top1[0])
    i_bottom2 = np.where(x == bottom2)
    i_bottom2 = int(i_bottom2[0])
    i_top2 = np.where(x == top2)
    i_top2 = int(i_top2[0])
    norm_fft = normalize(fft)
    mean1 = np.mean(np.absolute(norm_fft[i_bottom1:i_top1]))
    mean2 = np.mean(np.absolute(norm_fft[i_bottom2:i_top2]))
    return mean1/mean2

def get_means_ratio(audio_path, bottom1, top1, bottom2, top2):
    labels = os.listdir(audio_path)
    all_means_ratio = []
    for label in labels:
        fft = get_fft(audio_path, label)
        mean_ratio = get_mean_ratio(fft, bottom1, top1, bottom2, top2)
        all_means_ratio.append(mean_ratio)
    return np.array(all_means_ratio)

def get_means(audio_path, bottom, top):
    labels = os.listdir(audio_path)
    all_means = []
    for label in labels:
        fft = get_fft(audio_path, label)
        mean = get_mean(fft, bottom, top)
        all_means.append(mean)
    return np.array(all_means)

def get_means_ratio_dist(audio_path1, audio_path2, bottom1, top1, bottom2, top2, title, label1, label2):
    means1 = get_means_ratio(audio_path1, bottom1, top1, bottom2, top2)
    means1 = pd.Series(means1)
    plt.hist(means1, bins=np.arange(means1.min(), means1.max() + 1, 0.5), alpha=0.7, label=label1)
    means2 = get_means_ratio(audio_path2, bottom1, top1, bottom2, top2)
    means2 = pd.Series(means2)
    plt.hist(means2, bins=np.arange(means2.min(), means2.max() + 1, 0.5), alpha=0.7, label=label2)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('means ratio amplitude')
    plt.ylabel('frequency')
    plt.show()

def get_means_dist(audio_path1, audio_path2, bottom, top, title, label1, label2):
    means1 = get_means(audio_path1, bottom, top)
    means1 = pd.Series(means1)
    plt.hist(means1, bins=np.arange(means1.min(), means1.max() + 1, 0.05), alpha=0.7, label=label1)
    means2 = get_means(audio_path2, bottom, top)
    means2 = pd.Series(means2)
    plt.hist(means2, bins=np.arange(means2.min(), means2.max() + 1, 0.05), alpha=0.7, label=label2)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('mean amplitude')
    plt.ylabel('frequency')
    plt.show()
    #print(stds.value_counts())
    #return np.histogram(stds)


def get_two_norm_f0s(filepath1, filepath2, logf0s_normalization):
    log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B = get_f0s(logf0s_normalization)
    f01, _, _, _ = decompose(filepath1)
    f01_normalized = normalize(f01, math.exp(log_f0s_mean_A), math.exp(log_f0s_std_A))
    print(math.exp(log_f0s_mean_A))
    print(math.exp(log_f0s_std_A))
    f02, _, _, _ = decompose(filepath2)
    print(math.exp(log_f0s_mean_B))
    print(math.exp(log_f0s_std_B))
    f02_normalized = normalize(f02, math.exp(log_f0s_mean_B), math.exp(log_f0s_std_B))
    print(np.mean(f01))
    print(np.mean(f02))
    print(np.mean(f01_normalized))
    print(np.mean(f02_normalized))

def avg_fft(audio_path):
    labels = os.listdir(audio_path)
    fft = get_fft(audio_path, labels[0])
    #dur = librosa.get_duration(samples, sr=8000)  # = 24000/8000 = 3
    avg_fft = fft
    for label in labels[1:]:
        fft = get_fft(audio_path, label)
        avg_fft = np.array(avg_fft) + np.array(fft)
    avg_fft = avg_fft / len(labels)
    return avg_fft

def get_fft(audio_path, label):
    samples, sample_rate = librosa.load(audio_path + '/' + label, sr=8000)
    samples = np.resize(samples, 24000)
    plot(samples,'123')
    fft = scipy.fft.fft(samples)
    plot(fft, 'fft')
    return fft

def get_stds(audio_path):
    labels = os.listdir(audio_path)
    all_stds = []
    for label in labels:
        fft = get_fft(audio_path, label)
        std = np.std(fft)
        all_stds.append(std)
    return np.array(all_stds)

def get_std_frequency_dist(audio_path1, audio_path2, title, label1, label2):
    stds1 = get_stds(audio_path1)
    stds1 = pd.Series(stds1)
    plt.hist(stds1, bins=np.arange(stds1.min(), stds1.max() + 1), alpha=0.7, label=label1)
    stds2 = get_stds(audio_path2)
    stds2 = pd.Series(stds2)
    plt.hist(stds2, bins=np.arange(stds2.min(), stds2.max() + 1), alpha=0.7, label=label2)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Std')
    plt.ylabel('frequency')
    plt.show()
    #print(stds.value_counts())
    #return np.histogram(stds)

def fft_show(audio_path, title):
    samples, sample_rate = librosa.load(audio_path, sr=8000)
    fft = scipy.fft.fft(samples)
    plot(fft, title)

def get_x(fft):
    sample_rate = 8000
    duration = 3
    n = sample_rate * duration
    x = scipy.fft.fftfreq(n, 1 / sample_rate)
    half_len = int(len(x) / 2)
    x = x[:half_len]
    fft = fft[:half_len]
    return x, fft

def plot(fft, title):
    x, fft = get_x(fft)
    plt.plot(x, fft)
    # Add title and axis names
    plt.title(title)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    logf0s_normalization = '/tmp/pycharm_project_279/cache_interspeech/logf0s_normalization.npz'
    filepath1 = '/tmp/pycharm_project_279/converted_sound/mix_to_teller/'
    filepath2 = '/tmp/pycharm_project_279/converted_sound/mix_to_mrs/arisotle_2.wav'
    #fft_show(filepath1, 'aristotle teller')
    #fft_show(filepath2, 'aristotle mrs bennet')
    filepath1 = '/tmp/pycharm_project_279/converted_sound/mix_to_interview/arisotle_2.wav'
    filepath2 = '/tmp/pycharm_project_279/converted_sound/mix_to_speech/arisotle_2.wav'
    #fft_show(filepath1, 'aristotle interview')
    #fft_show(filepath2, 'aristotle speech')

    filepath1 = '/tmp/pycharm_project_279/converted_sound/mix_to_interview'
    filepath2 = '/tmp/pycharm_project_279/converted_sound/mix_to_speech'
    #get_std_frequency_dist(filepath1, filepath2, 'std of interview vs speech converted', 'Interview', 'speech')
    #get_means_dist(filepath1, filepath2, 0, 550, 'amplitudes of 0-550 hz', 'interview', 'speech')

    #get_means_ratio_dist(filepath1, filepath2, 0, 500, 900, 1300, 'amplitudes of 0-500 / 900-1300 hz', 'interview', 'speech')
    filepath3 = '/tmp/pycharm_project_279/converted_sound/mix_to_teller'
    filepath4 = '/tmp/pycharm_project_279/converted_sound/mix_to_mrs'
    get_means_ratio_dist(filepath3, filepath4, 50, 500, 1000, 1300, 'amplitudes of 50-500 / 1000-1300 hz', 'teller',
                         'mrs Bennet')

    #get_std_frequency_dist(filepath3, filepath4, 'std of teller vs mrs bennet converted', 'teller', 'mrs bennet')
    #get_std_frequency_dist(filepath1, filepath3, 'std of converted speech vs interview', 'interview to speech', 'speech to interview')
    #get_std_frequency_dist(filepath2, filepath3, 'std of speech vs interview', 'interview', 'speech')

    #avg1 = avg_fft('/tmp/pycharm_project_279/converted_sound/mix_to_teller')
    #plot(avg1, 'avg teller')
    #avg2 = avg_fft('/tmp/pycharm_project_279/converted_sound/mix_to_mrs')
    #plot(avg2, 'avg mrs')
    #avg3 = avg_fft('/tmp/pycharm_project_279/converted_sound/mix_to_interview')
    #plot(avg3, 'avg interview')
    #avg4 = avg_fft('/tmp/pycharm_project_279/converted_sound/mix_to_speech')
    #plot(avg4, 'avg speech')

    #plot(avg3, 'Mean converted interview fft')
    #plot(avg4, 'Mean converted speech fft')
    #fft_show(filepath2)
    #get_two_norm_f0s(filepath1, filepath2, logf0s_normalization)
    #compare_two()
    #spec_mrs = avg_spectrogram(audio_path,'mrs_bennet')
    #spec_mr = avg_spectrogram(audio_path, 'mr_bennet')
    #plot_spectrogram(np.abs(spec_mrs-spec_mr))
    #spec_jane = avg_spectrogram(audio_path, 'jane')
    #spec_teller = avg_spectrogram(audio_path, 'teller')
    #plot_spectrogram(np.abs(spec_teller - spec_jane))