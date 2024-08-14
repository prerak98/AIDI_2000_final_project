import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def plot_waveform(file_name):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_name, sr=None)

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Save the plot as a PNG file
    plt.savefig('waveform_plot.png')
    print("Waveform plot saved as 'waveform_plot.png'")

    # Optional: Display the plot
    # plt.show()  # Comment this out if not needed

# Example usage
audio_dir = '../data/ravdess/Actor_01/'  # Example path
file_name = os.path.join(audio_dir, '03-01-01-01-01-01-01.wav')
plot_waveform(file_name)
