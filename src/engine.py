import base64
import requests
import os


class LlamaChatEngine:
    def __init__(
        self,
        server_url=os.environ.get("LLAMA_SERVER_URL", "127.0.0.1:8080"),
        system_prompt="Du bist ein hilfreicher KI-Assistent.",
        model_name=os.environ.get(
            "MODEL_NAME",
            "gemma-4-E2B-it-GGUF",  # Name ist meist egal beim llama.cpp server
        ),
    ):
        """
        Initialisiert die Chat-Engine und legt den System-Prompt fest.
        """
        self.server_url = server_url
        self.endpoint = f"{self.server_url}/v1/chat/completions"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.messages: list[dict[str, str | list[dict]]] = []
        self.reset_chat()  # Setzt den Chatverlauf initial auf (inkl. System-Prompt)

    def reset_chat(self):
        """Löscht den bisherigen Verlauf und startet neu mit dem System-Prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("--- Chatverlauf wurde zurückgesetzt ---")

    def _encode_audio(self, audio_path):
        """Hilfsfunktion: Liest Audio ein und wandelt es in Base64 um."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Die Datei '{audio_path}' wurde nicht gefunden.")

        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        audio_format = audio_path.split(".")[-1].lower()
        if audio_format not in ["wav", "mp3"]:
            audio_format = "wav"

        return base64_audio, audio_format

    def send_message(self, text=None, audio_path=None, temperature=0.4):
        """
        Fügt die neue Nachricht (Text und/oder Audio) dem Verlauf hinzu,
        sendet den gesamten Verlauf an die API und speichert die Antwort.
        """
        if not text and not audio_path:
            return "Fehler: Es muss entweder Text oder Audio übergeben werden."

        # 1. Inhalt der neuen Benutzernachricht zusammenbauen
        message_content: list[dict[str, str | dict]] = []

        # Falls Text vorhanden ist, hinzufügen
        if text:
            message_content.append({"type": "text", "text": text})

        # Falls Audio vorhanden ist, verarbeiten und hinzufügen
        if audio_path:
            try:
                b64_audio, a_format = self._encode_audio(audio_path)
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": a_format},
                    }
                )

                message_content.append(
                    {
                        "type": "text",
                        "text": "The user just spoke to you (audio). Respond to what they said.",
                    }
                )
            except Exception as e:
                return f"Audio-Fehler: {e}"

        # 2. Nachricht an unseren lokalen Verlauf anhängen
        self.messages.append({"role": "user", "content": message_content})

        # 3. Payload für den Server erstellen (mit gesamtem Verlauf)
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": temperature,
        }

        # 4. API Request senden
        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()

            result_json = response.json()

            # Die exakte Antwort des Assistenten extrahieren
            assistant_message = result_json["choices"][0]["message"]

            # 5. Die Antwort des Servers an unseren Verlauf anhängen!
            # Dadurch "erinnert" sich das Modell bei der nächsten Frage an seine eigene Antwort.
            self.messages.append(assistant_message)

            return assistant_message.get("content", "")

        except requests.exceptions.RequestException as e:
            # Bei einem Verbindungsfehler entfernen wir unsere letzte Nachricht wieder,
            # damit wir es noch einmal versuchen können, ohne den Chatverlauf zu vergiften.
            self.messages.pop()
            return f"API-Kommunikationsfehler: {e}\nDetails: {getattr(response, 'text', 'Keine Details')}"


# ==========================================
# Beispiel für die Nutzung der Engine
# ==========================================
if __name__ == "__main__":
    # Engine instanziieren
    engine = LlamaChatEngine(
        system_prompt="Du bist ein freundlicher Assistent, der Audios präzise analysiert und kurze Antworten gibt."
    )

    # 1. Turn: Audio mitsenden und etwas dazu fragen
    print("User: Bitte transkribiere und fasse zusammen, was hier gesagt wird.")
    antwort1 = engine.send_message(
        text="Bitte transkribiere und fasse zusammen, was hier gesagt wird.",
        audio_path="beispiel.wav",  # <-- Hier Pfad zu deiner Datei anpassen
    )
    print(f"KI: {antwort1}\n")

    # 2. Turn: Eine Nachfrage stellen (Ohne Audio, aber die KI kennt den Kontext noch!)
    print("User: Welche Emotion hatte der Sprecher in dieser Aufnahme?")
    antwort2 = engine.send_message(
        text="Welche Emotion hatte der Sprecher in dieser Aufnahme?"
    )
    print(f"KI: {antwort2}\n")

    # 3. Turn: Ein komplett neues Audio mitgeben
    print("User: Und was unterscheidet dieses zweite Audio vom ersten?")
    antwort3 = engine.send_message(
        text="Und was unterscheidet dieses zweite Audio vom ersten?",
        audio_path="zweites_beispiel.wav",
    )
    print(f"KI: {antwort3}\n")

    # Den aktuellen kompletten Chatverlauf ansehen
    # print(engine.messages)
