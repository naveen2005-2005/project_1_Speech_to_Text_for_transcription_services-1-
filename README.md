# project_1_Speech_to_Text_for_transcription_services-1-
📦 Requirements
The notebook installs key packages:

bash
Copy
Edit
pip install kaggle
pip install transformers torchaudio librosa noisereduce
pip install openai-whisper        # Optional if using Whisper
pip install datasets
pip install jiwer                 # For Word Error Rate (WER) calculations
📁 Data Upload
python
Copy
Edit
from google.colab import files
files.upload()  # Upload your .tar.gz dataset or audio file
🧹 Module 1: Data Cleaning
Loads an audio file using librosa

Trims silence and normalizes volume

Displays the waveform for visual verification

python
Copy
Edit
y, sr = librosa.load('/sp01_street_sn5.wav', sr=None)
y_clean, _ = librosa.effects.trim(y)
y_normalized = librosa.util.normalize(y_clean)
📊 Module 2: Data Analysis
Computes and displays the spectrogram of the cleaned audio

Helps visually inspect noise, clarity, and signal quality

python
Copy
Edit
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_normalized)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
📈 Module 3: Visualization Tools
Calculates Word Error Rate (WER) using jiwer:

Compares predicted transcription against actual text

Useful for measuring transcription model accuracy

python
Copy
Edit
from jiwer import wer
wer_score = wer(actual_transcription, predicted_transcription)
Also includes a small bar chart visualization comparing WER across test cases.

