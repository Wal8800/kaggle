import librosa
import matplotlib.pyplot as plt
import librosa.display

from process_audio_data import padded_2d_array


class MFCCBuilder:
    def __init__(self, frame_weight=80, frame_shift=10, sampling_rate=44100, n_mfcc=64, audio_duration=2):
        self.n_mfcc = n_mfcc
        self.n_fft = int(frame_weight / 1000 * sampling_rate)
        self.sampling_rate = sampling_rate
        self.hop_length = int(frame_shift / 1000 * sampling_rate)
        self.audio_duration = audio_duration
        self.frame_weight = frame_weight
        self.frame_shift = frame_shift

    def generate_log_mfcc(self, directory, fname):
        y, sr = librosa.load(directory + fname, sr=self.sampling_rate)
        s = librosa.feature.mfcc(y=y,
                                 sr=self.sampling_rate,
                                 n_mfcc=self.n_mfcc,
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length)
        return librosa.power_to_db(s)

    def generate_padded_log_mfcc(self, directory, fname):
        mfccs = self.generate_log_mfcc(directory, fname)
        input_frame_length = int(self.audio_duration * 1000 / self.frame_shift)
        return padded_2d_array(mfccs, input_frame_length)

    def visualise_mfcc(self, mfccs):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time',
                                 sr=self.sampling_rate,
                                 hop_length=self.hop_length)
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()


def process_and_save_mfccs(row):
    fname = row[1][0]
    builder = MFCCBuilder()
    padded_melspec = builder.generate_padded_log_mfcc("data/train_curated/", fname)
    padded_melspec.dump("processed/mfccs/" + fname + ".pickle")
