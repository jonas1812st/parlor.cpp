import sounddevice as sd
import soundfile as sf
import numpy as np
import queue

import sys
import os

# Importiere unsere Engine aus der anderen Datei
from chat_engine import LlamaChatEngine


def record_audio_from_mic(filename="temp_user_input.wav", samplerate=44100):
    """
    Nimmt Audio über das Standard-Mikrofon auf.
    Die Aufnahme startet beim ersten Tastendruck (Enter) und stoppt beim zweiten.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """Wird kontinuierlich aufgerufen und schiebt Audio-Daten in die Queue."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n" + "=" * 50)
    input("🎤 Drücke [ENTER], um die Sprachaufnahme zu STARTEN...")
    print("🔴 Aufnahme läuft... (Spreche jetzt. Drücke erneut [ENTER], um zu STOPPEN)")

    # Audiostream öffnen (1 Kanal = Mono, 16kHz reicht für Sprache optimal aus)
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        input()  # Blockiert, bis der Nutzer wieder Enter drückt

    print("⏳ Verarbeite Audio...")

    # Alle Daten aus der Queue holen und zusammensetzen
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())

    audio_data = np.concatenate(audio_data, axis=0)

    # Als WAV-Datei abspeichern
    sf.write(filename, audio_data, samplerate)
    return filename


def main():
    # Chat Engine initialisieren
    print("🤖 Starte LlamaChatEngine...")
    chat_engine = LlamaChatEngine(
        server_url="http://192.168.80.7:8080",
        system_prompt=(
            "Du bist ein freundlicher konversationeller KI Assistent. Der Benutzer spricht mit dir "
            "durch ein Mikrofon. Du musst IMMER das respond_to_user Tool nutzen, um zu antworten. "
            "Transkribiere zuerst, was der Benutzer sagt und verfasse danach deine Antwort."
            # "Du bist ein KI-Übersetzer. Der Benutzer spricht mit dir durch ein Mikrofon. "
            # "Transkribiere zuerst, was der Benutzer sagt und verfasse danach eine Übersetzung. "
            # "Deine Zielsprache ist Englisch."
        ),
        save_messages=True,
        choose_tool="respond_to_user",
    )

    start_msg = "Hallo! Ich bin bereit. Folge den Anweisungen, um mit mir zu sprechen."
    print(f"\nKI: {start_msg}")

    audio_file = "./samples/temp-user-input.wav"

    try:
        while True:
            record_audio_from_mic(filename=audio_file)
            print("🧠 Sende Audio an Llama.cpp (Modell denkt nach)...")
            ergebnis = chat_engine.send_message(audio_path=audio_file)

            if isinstance(ergebnis, dict) and "transcription" in ergebnis:
                # Erfolgreicher Tool-Call / JSON-Format
                transkription = ergebnis.get("transcription", "")
                antwort = ergebnis.get("response", "")

                print(f"\n🗣️  Du hast gesagt: '{transkription}'")
                print(f"🤖 KI Antwortet:  '{antwort}'")
            else:
                print(f"\n⚠️ Unerwartetes Format vom Server erhalten: {ergebnis}")

    except KeyboardInterrupt:
        # Beenden mit Strg+C
        print("\n\n👋 Chatbot wird beendet. Auf Wiedersehen!")
        if os.path.exists(audio_file):
            os.remove(audio_file)  # Räumt die temporäre Audiodatei auf
        sys.exit(0)


if __name__ == "__main__":
    main()
