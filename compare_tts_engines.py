# compare_tts_engines.py

import os
import time
import matplotlib.pyplot as plt
import librosa
import numpy as np
from pydub import AudioSegment
import tempfile

from TTS.api import TTS
from lib.classes.tts_engines.elevenlabs import ElevenLabs

# Sample text
test_text = "This is a test to compare TTS performance and quality."

def save_audio_from_tts(tts, text, output_path):
    try:
        tts.tts_to_file(text=text, file_path=output_path)
    except Exception as e:
        print(f"Error generating audio: {e}")

def safe_get_audio_stats(file):
    if not file or not os.path.isfile(file):
        print(f"Skipping audio stats: invalid file: {file}")
        return None
    try:
        ext = os.path.splitext(file)[-1].lower()
        if ext == '.mp3':
            audio = AudioSegment.from_mp3(file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                audio.export(tmp_wav.name, format="wav")
                y, sr = librosa.load(tmp_wav.name, sr=None)
            os.unlink(tmp_wav.name)
        else:
            y, sr = librosa.load(file, sr=None)

        # Basic stats
        rms = np.sqrt(np.mean(y**2))
        peak = np.max(np.abs(y))
        duration = librosa.get_duration(y=y, sr=sr)

        # Extra audio features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        return {
            "rms": rms, "peak": peak, "duration": duration,
            "zcr": zcr, "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff
        }
    except Exception as e:
        print(f"Error extracting audio stats for {file}: {e}")
        return None

# # --- Coqui (external TTS lib) ---
# print("Running Coqui TTS...")
# coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
# coqui_output = "coqui_output.wav"
# start_coqui = time.time()
# save_audio_from_tts(coqui_tts, test_text, coqui_output)
# end_coqui = time.time()
# coqui_time = end_coqui - start_coqui
# coqui_size_kb = os.path.getsize(coqui_output) / 1024 if os.path.exists(coqui_output) else 0
# coqui_stats = safe_get_audio_stats(coqui_output)

# # --- ElevenLabs (your new class) ---
# print("Running ElevenLabs TTS...")
# elevenlabs = ElevenLabs(None)
# start_eleven = time.time()
# eleven_output = elevenlabs.convert(test_text)
# end_eleven = time.time()
# eleven_time = end_eleven - start_eleven
# eleven_size_kb = os.path.getsize(eleven_output) / 1024 if os.path.exists(eleven_output) else 0
# eleven_stats = safe_get_audio_stats(eleven_output)

# # --- Summary Table ---
# print("\n=== TTS Comparison ===")
# print(f"{'Engine':<15}{'Time (s)':<12}{'Size (KB)':<12}{'Duration(s)':<15}")
# print(f"{'Coqui':<15}{coqui_time:<12.2f}{coqui_size_kb:<12.1f}{coqui_stats['duration'] if coqui_stats else '-':<15}")
# print(f"{'ElevenLabs':<15}{eleven_time:<12.2f}{eleven_size_kb:<12.1f}{eleven_stats['duration'] if eleven_stats else '-':<15}")

# print("\n=== Extended Audio Stats ===")
# print(f"{'Engine':<15}{'RMS':<12}{'Peak':<12}{'ZCR':<12}{'Centroid':<12}{'Bandwidth':<12}{'Rolloff':<12}")
# print(f"{'Coqui':<15}{coqui_stats['rms'] if coqui_stats else '-':<12.4f}{coqui_stats['peak'] if coqui_stats else '-':<12.4f}"
#       f"{coqui_stats['zcr'] if coqui_stats else '-':<12.4f}{coqui_stats['centroid'] if coqui_stats else '-':<12.1f}"
#       f"{coqui_stats['bandwidth'] if coqui_stats else '-':<12.1f}{coqui_stats['rolloff'] if coqui_stats else '-':<12.1f}")
# print(f"{'ElevenLabs':<15}{eleven_stats['rms'] if eleven_stats else '-':<12.4f}{eleven_stats['peak'] if eleven_stats else '-':<12.4f}"
#       f"{eleven_stats['zcr'] if eleven_stats else '-':<12.4f}{eleven_stats['centroid'] if eleven_stats else '-':<12.1f}"
#       f"{eleven_stats['bandwidth'] if eleven_stats else '-':<12.1f}{eleven_stats['rolloff'] if eleven_stats else '-':<12.1f}")

# # --- Plotting: Line Graphs ---
# labels = ['Coqui', 'ElevenLabs']
# x = np.arange(len(labels))

# fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# # 1. Time & Size
# axs[0].plot(x, [coqui_time, eleven_time], marker='o', label='Time (s)', color='blue')
# axs[0].plot(x, [coqui_size_kb, eleven_size_kb], marker='o', label='Size (KB)', color='green')
# axs[0].set_xticks(x)
# axs[0].set_xticklabels(labels)
# axs[0].legend()
# axs[0].set_title('Time & File Size')

# # 2. RMS, Peak, ZCR
# axs[1].plot(x, [coqui_stats['rms'], eleven_stats['rms']], marker='o', label='RMS', color='purple')
# axs[1].plot(x, [coqui_stats['peak'], eleven_stats['peak']], marker='o', label='Peak', color='red')
# axs[1].plot(x, [coqui_stats['zcr'], eleven_stats['zcr']], marker='o', label='ZCR', color='orange')
# axs[1].set_xticks(x)
# axs[1].set_xticklabels(labels)
# axs[1].legend()
# axs[1].set_title('RMS, Peak, ZCR')

# # 3. Spectral Features
# axs[2].plot(x, [coqui_stats['centroid'], eleven_stats['centroid']], marker='o', label='Centroid', color='cyan')
# axs[2].plot(x, [coqui_stats['bandwidth'], eleven_stats['bandwidth']], marker='o', label='Bandwidth', color='magenta')
# axs[2].plot(x, [coqui_stats['rolloff'], eleven_stats['rolloff']], marker='o', label='Rolloff', color='yellow')
# axs[2].set_xticks(x)
# axs[2].set_xticklabels(labels)
# axs[2].legend()
# axs[2].set_title('Spectral Features')

# plt.tight_layout()
# plt.savefig("tts_comparison.png")
# print("\nLine plot saved as: tts_comparison.png")

# --- Pre-existing audio file (sample.mp3) ---
print("Loading pre-existing sample audio (sample.mp3) for comparison...")
sample_audio_path = os.path.join("comparison", "sample.mp3")
start_sample = time.time()
# No generation step needed, just measuring file size and stats
sample_size_kb = os.path.getsize(sample_audio_path) / 1024 if os.path.exists(sample_audio_path) else 0
sample_stats = safe_get_audio_stats(sample_audio_path)
end_sample = time.time()
sample_time = end_sample - start_sample  # Only measuring load time, not generation

# --- ElevenLabs (your new class) ---
print("Running ElevenLabs TTS...")
elevenlabs = ElevenLabs(None)
start_eleven = time.time()
eleven_output = elevenlabs.convert(test_text)
end_eleven = time.time()
eleven_time = end_eleven - start_eleven
eleven_size_kb = os.path.getsize(eleven_output) / 1024 if os.path.exists(eleven_output) else 0
eleven_stats = safe_get_audio_stats(eleven_output)

# --- Summary Table ---
print("\n=== TTS Comparison ===")
print(f"{'Engine':<15}{'Time (s)':<12}{'Size (KB)':<12}{'Duration(s)':<15}")
print(f"{'Coqui':<15}{sample_time:<12.2f}{sample_size_kb:<12.1f}{sample_stats['duration'] if sample_stats else '-':<15}")
print(f"{'ElevenLabs':<15}{eleven_time:<12.2f}{eleven_size_kb:<12.1f}{eleven_stats['duration'] if eleven_stats else '-':<15}")

print("\n=== Extended Audio Stats ===")
print(f"{'Engine':<15}{'RMS':<12}{'Peak':<12}{'ZCR':<12}{'Centroid':<12}{'Bandwidth':<12}{'Rolloff':<12}")
print(f"{'Coqui':<15}{sample_stats['rms'] if sample_stats else '-':<12.4f}{sample_stats['peak'] if sample_stats else '-':<12.4f}"
      f"{sample_stats['zcr'] if sample_stats else '-':<12.4f}{sample_stats['centroid'] if sample_stats else '-':<12.1f}"
      f"{sample_stats['bandwidth'] if sample_stats else '-':<12.1f}{sample_stats['rolloff'] if sample_stats else '-':<12.1f}")
print(f"{'ElevenLabs':<15}{eleven_stats['rms'] if eleven_stats else '-':<12.4f}{eleven_stats['peak'] if eleven_stats else '-':<12.4f}"
      f"{eleven_stats['zcr'] if eleven_stats else '-':<12.4f}{eleven_stats['centroid'] if eleven_stats else '-':<12.1f}"
      f"{eleven_stats['bandwidth'] if eleven_stats else '-':<12.1f}{eleven_stats['rolloff'] if eleven_stats else '-':<12.1f}")

# --- Plotting: Line Graphs ---
labels = ['Coqui', 'ElevenLabs']
x = np.arange(len(labels))

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# 1. Time & Size
axs[0].plot(x, [sample_time, eleven_time], marker='o', label='Time (s)', color='blue')
axs[0].plot(x, [sample_size_kb, eleven_size_kb], marker='o', label='Size (KB)', color='green')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].legend()
axs[0].set_title('Time & File Size')

# 2. RMS, Peak, ZCR
axs[1].plot(x, [sample_stats['rms'], eleven_stats['rms']], marker='o', label='RMS', color='purple')
axs[1].plot(x, [sample_stats['peak'], eleven_stats['peak']], marker='o', label='Peak', color='red')
axs[1].plot(x, [sample_stats['zcr'], eleven_stats['zcr']], marker='o', label='ZCR', color='orange')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].legend()
axs[1].set_title('RMS, Peak, ZCR')

# 3. Spectral Features
axs[2].plot(x, [sample_stats['centroid'], eleven_stats['centroid']], marker='o', label='Centroid', color='cyan')
axs[2].plot(x, [sample_stats['bandwidth'], eleven_stats['bandwidth']], marker='o', label='Bandwidth', color='magenta')
axs[2].plot(x, [sample_stats['rolloff'], eleven_stats['rolloff']], marker='o', label='Rolloff', color='yellow')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
axs[2].legend()
axs[2].set_title('Spectral Features')

plt.tight_layout()
plt.savefig("tts_comparison_1.png")
print("\nLine plot saved as: tts_comparison_1.png")