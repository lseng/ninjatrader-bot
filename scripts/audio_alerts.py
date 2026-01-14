#!/usr/bin/env python3
"""
Audio Alerts for Trading Signals

Uses Microsoft Edge Neural TTS for natural-sounding voice alerts.
Falls back to Windows SAPI if edge-tts unavailable.

Voices:
- en-US-AriaNeural: Warm, friendly female (default)
- en-US-GuyNeural: Friendly male
- en-US-JennyNeural: Professional female
- en-US-DavisNeural: Calm male
"""

import asyncio
import tempfile
import os
import time

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Available voices
VOICES = {
    'aria': 'en-US-AriaNeural',      # Warm female (default)
    'guy': 'en-US-GuyNeural',        # Friendly male
    'jenny': 'en-US-JennyNeural',    # Professional female
    'davis': 'en-US-DavisNeural',    # Calm male
}

DEFAULT_VOICE = 'aria'


def format_alert_text(signal: dict, include_levels: bool = False) -> str:
    """Format signal into spoken text - kept short and simple"""
    direction = signal.get('direction', 'unknown')
    confidence = signal.get('confidence', 0)
    entry = signal.get('entry', 0)

    # Format entry price for speech (e.g., 6939.43 -> "69 39 43")
    entry_str = f"{entry:.2f}".replace('.', ' ')

    # Direction, entry, confidence
    text = f"{direction}. {entry_str}. {confidence} percent."

    return text


async def _speak_edge_tts(text: str, voice: str = None, rate: str = "+15%"):
    """
    Speak using Edge TTS (high quality neural voices)

    Args:
        text: Text to speak
        voice: Voice name
        rate: Speech rate adjustment (e.g., "+15%", "+25%", "-10%")
    """
    import edge_tts
    import pygame

    voice = VOICES.get(voice or DEFAULT_VOICE, VOICES[DEFAULT_VOICE])

    # Create communicate with rate adjustment for natural pacing
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    tmp_file = os.path.join(tempfile.gettempdir(), 'trade_alert.mp3')

    await communicate.save(tmp_file)

    # Play with pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(tmp_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def speak_edge_tts(text: str, voice: str = None, rate: str = "+15%"):
    """Synchronous wrapper for edge TTS"""
    asyncio.run(_speak_edge_tts(text, voice, rate))


def speak_windows_sapi(text: str):
    """Fallback: Use Windows built-in speech"""
    import subprocess

    ps_script = f'''
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Rate = 1
    $synth.Speak("{text}")
    '''

    subprocess.run(
        ['powershell', '-Command', ps_script],
        capture_output=True,
        timeout=30
    )


def speak_alert(signal: dict, voice: str = None, use_edge: bool = True):
    """
    Speak a trade alert.

    Args:
        signal: Signal dict with direction, confidence, strategy, entry
        voice: Voice name (aria, guy, jenny, davis)
        use_edge: Use Edge TTS (True) or Windows SAPI (False)
    """
    text = format_alert_text(signal)

    try:
        if use_edge:
            speak_edge_tts(text, voice)
        else:
            speak_windows_sapi(text)
    except Exception as e:
        print(f"[Audio Error] {e}")
        # Try fallback
        try:
            speak_windows_sapi(text)
        except:
            # Last resort: just beep
            print('\a', end='', flush=True)


# Path to custom airport chime MP3
CHIME_FILE = os.path.join(os.path.dirname(__file__), 'airport_chime.mp3')


def play_chime(style: str = 'airport'):
    """
    Play a pleasant chime before alert.

    Styles:
    - 'airport': Custom airport PA chime MP3
    - 'simple': Two-tone beep
    - 'alert': Attention-getting rising tone
    """
    try:
        if style == 'airport' and os.path.exists(CHIME_FILE):
            # Play custom airport chime MP3
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(CHIME_FILE)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            return

        # Fallback to beep tones
        import winsound

        if style == 'airport':
            # Fallback: G5 - E5 - C5 (descending major triad)
            winsound.Beep(784, 180)  # G5
            time.sleep(0.05)
            winsound.Beep(659, 180)  # E5
            time.sleep(0.05)
            winsound.Beep(523, 250)  # C5
            time.sleep(0.15)

        elif style == 'simple':
            winsound.Beep(523, 150)  # C5
            winsound.Beep(659, 150)  # E5

        elif style == 'alert':
            winsound.Beep(440, 100)  # A4
            winsound.Beep(554, 100)  # C#5
            winsound.Beep(659, 150)  # E5

    except Exception as e:
        pass


def _play_signal_audio(text: str, voice: str, chime_style: str,
                       overlap_seconds: float, speech_rate: str):
    """Internal function to play audio - runs in thread to avoid async issues."""
    try:
        import pygame
        import edge_tts

        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Load chime
        chime_sound = pygame.mixer.Sound(CHIME_FILE)
        chime_duration = chime_sound.get_length()
        speech_delay = max(0, chime_duration - overlap_seconds)

        # Generate speech audio file
        voice_id = VOICES.get(voice, VOICES[DEFAULT_VOICE])
        tmp_file = os.path.join(tempfile.gettempdir(), f'trade_alert_{int(time.time()*1000)}.mp3')

        async def generate_speech():
            communicate = edge_tts.Communicate(text, voice_id, rate=speech_rate)
            await communicate.save(tmp_file)

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech())
        loop.close()

        # Play chime
        channel = pygame.mixer.Channel(0)
        channel.play(chime_sound)

        # Wait until overlap point
        time.sleep(speech_delay)

        # Play speech
        pygame.mixer.music.load(tmp_file)
        pygame.mixer.music.play()

        # Wait for both to finish
        while channel.get_busy() or pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except Exception as e:
        # Silent fail - just beep
        try:
            import winsound
            winsound.Beep(800, 300)
        except:
            print('\a', end='', flush=True)


def speak_signal(signal: dict, voice: str = None, chime: bool = True,
                 chime_style: str = 'airport', overlap_seconds: float = 3.0,
                 speech_rate: str = "+15%"):
    """
    Full alert with optional chime + voice, with seamless overlap.
    Runs in a separate thread to avoid blocking async event loops.
    """
    import threading

    text = format_alert_text(signal)
    voice = voice or DEFAULT_VOICE

    if chime and chime_style == 'airport' and os.path.exists(CHIME_FILE):
        # Run audio in thread to avoid asyncio conflicts
        thread = threading.Thread(
            target=_play_signal_audio,
            args=(text, voice, chime_style, overlap_seconds, speech_rate),
            daemon=True
        )
        thread.start()
    else:
        # No chime - just speak
        try:
            speak_edge_tts(text, voice, speech_rate)
        except:
            pass


# Test function
def test_voices():
    """Test all available voices"""
    test_signal = {
        'direction': 'LONG',
        'confidence': 85,
        'strategy': 'ORDER_BLOCK',
        'entry': 6950.25
    }

    for name in VOICES:
        print(f"Testing {name}...")
        speak_signal(test_signal, voice=name)
        time.sleep(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test audio alerts')
    parser.add_argument('--voice', choices=list(VOICES.keys()), default='aria')
    parser.add_argument('--test-all', action='store_true', help='Test all voices')
    parser.add_argument('--text', type=str, help='Custom text to speak')
    args = parser.parse_args()

    if args.test_all:
        test_voices()
    elif args.text:
        speak_edge_tts(args.text, args.voice)
    else:
        # Default test
        signal = {
            'direction': 'LONG',
            'confidence': 92,
            'strategy': 'SMC_CONFLUENCE',
            'entry': 6950.50
        }
        print(f"Testing {args.voice} voice...")
        speak_signal(signal, voice=args.voice)
        print("Done!")
