from lib.classes.tts_engines.elevenlabs import ElevenLabs

# Create a dummy session (you may not need any session logic for ElevenLabs)
session = {}
tts_engine = ElevenLabs(session)

# Test text
test_text = "Hello, this is a test of the ElevenLabs TTS engine integration."

# Convert and generate audio
output_file = tts_engine.convert(test_text)

print(f"Generated audio file: {output_file}")