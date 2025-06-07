import requests

class ElevenLabs:
    """
    ElevenLabs Text-to-Speech engine integration.

    Provides an interface to convert text to speech using the ElevenLabs API
    with optional voice settings.
    """

    def __init__(self, session=None):
        """
        Initialize the ElevenLabs TTS engine.

        Args:
            session: Optional session parameter (not used here but kept for consistency with other engines).
        """
        self.api_key = "sk_b213633177a37d8403155b41aa08b8731c202fd880a684ac"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Voice: Rachel

    def convert(self, text, output_file="output.mp3", **kwargs):
        """
        Converts text to speech using the ElevenLabs API and saves it to a file.

        Args:
            text (str): The text to synthesize.
            output_file (str): Path to the output audio file. Defaults to 'output.mp3'.
            **kwargs: Optional voice settings, e.g., 'stability' and 'similarity_boost'.

        Returns:
            str: Path to the generated audio file.

        Raises:
            RuntimeError: If the API call fails or encounters errors.
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Build the payload with default or user-specified voice settings
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": kwargs.get("stability", 0.75),
                "similarity_boost": kwargs.get("similarity_boost", 0.75)
            }
        }

        try:
            # Stream response for large audio data
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=15) as response:
                if response.status_code == 200:
                    with open(output_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return output_file
                else:
                    try:
                        # Try to parse error details from JSON response
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    raise RuntimeError(f"TTS request failed (Status {response.status_code}): {error_detail}")

        except requests.RequestException as e:
            raise RuntimeError(f"Error communicating with ElevenLabs API: {e}")
