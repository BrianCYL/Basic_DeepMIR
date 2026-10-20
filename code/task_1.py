import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10)) 
    axes = axes.flatten()
    file = ['./dataset/nsynth-test/audio/bass_electronic_018-023-075.wav', './dataset/nsynth-test/audio/bass_electronic_018-050-075.wav', './dataset/nsynth-test/audio/bass_electronic_018-063-075.wav',
            './dataset/nsynth-test/audio/guitar_acoustic_010-022-075.wav', './dataset/nsynth-test/audio/guitar_acoustic_010-055-075.wav', './dataset/nsynth-test/audio/guitar_acoustic_010-068-075.wav',
            './dataset/nsynth-test/audio/keyboard_acoustic_004-028-075.wav', './dataset/nsynth-test/audio/keyboard_acoustic_004-051-075.wav', './dataset/nsynth-test/audio/keyboard_acoustic_004-062-075.wav']
    for i in range(9):
        instr = file[i].split('/')[-1].split('_')[0]
        pitch = file[i].split('/')[-1][-11:-8]
        data, sr = librosa.load(file[i])
        data = librosa.feature.melspectrogram(y=data, sr=sr)
        S_dB = librosa.power_to_db(data, ref=np.max)
        img = librosa.display.specshow(S_dB, ax=axes[i], x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        axes[i].set_title(f'Mel-frequency spectrogram of {instr} and pitch {pitch}')
        axes[i].label_outer()
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
    fig.tight_layout()
    if not os.path.exists('./result_image'):
        os.makedirs('./result_image')
    plt.savefig(f'./result_image/outcome.pdf')

if __name__ == '__main__':
    main()