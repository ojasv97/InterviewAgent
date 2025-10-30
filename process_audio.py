import os
import wave
from vosk import Model, KaldiRecognizer

# Path to the folder containing WAV files
AUDIO_FOLDER = "audio_uploads"
# Path to Vosk model folder (download from https://alphacephei.com/vosk/models)
MODEL_PATH = "C:\\Users\\A5815\OneDrive - Axtria\\Desktop\\Codes\\InterviewAgent\\models\\vosk-model-small-en-in-0.4"

# Load Vosk model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}")

model = Model(MODEL_PATH)

# Function to process a single WAV file
def transcribe_wav(file_path):
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Vosk requires WAV with 1 channel, 16-bit PCM")
    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(rec.Result())
    results.append(rec.FinalResult())
    wf.close()

    # Combine text from all results
    full_text = " ".join([eval(r)["text"] for r in results])
    return full_text

# Process all WAV files in folder
for filename in os.listdir(AUDIO_FOLDER):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(AUDIO_FOLDER, filename)
        try:
            text = transcribe_wav(file_path)
            print(f"\nFile: {filename}\nTranscription: {text}\n{'-'*50}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")