import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
import openai
import os
import asyncio
import websockets
import base64
import json
import sounddevice as sd
import numpy as np
import threading
import queue
import ssl
import certifi
# near the top of your file, with your other imports:
import sys

def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, whether running
    in development or as a PyInstaller bundle.
    """
    if getattr(sys, "_MEIPASS", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(__file__)
    return os.path.join(base, relative_path)

# --- Configuration ---
# Fetches the API key from environment variables.
# You MUST set this environment variable for the code to work.
API_KEY = os.environ.get("OPENAI_API_KEY") 
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# Language configuration
TARGET_LANGUAGES = {
    "Spanish": "es",
    "Swahili": "sw",
    "Burmese": "my",
    "Pashto": "ps"
}

# Audio configuration (must match Realtime API requirements)
SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = int(SAMPLE_RATE * 0.1) # 100ms of audio per block

# UI Configuration - Shoals color scheme
COLORS = {
    'bg': '#000000',           # Black background
    'primary': '#0a0a0a',      # Very dark gray
    'secondary': '#1a1a1a',    # Dark gray
    'accent': '#7EE068',       # Shoals green
    'success': '#7EE068',      # Green for start
    'danger': '#ff4444',       # Red for stop
    'text': '#ffffff',         # White text
    'text_light': '#cccccc',   # Light gray text
    'border': '#333333'        # Subtle borders
}

# --- Worker Threads (No Changes Here) ---

class TranscriptionWorker(threading.Thread):
    def __init__(self, transcript_queue, status_queue):
        super().__init__()
        self.transcript_queue = transcript_queue
        self.status_queue = status_queue
        self._stop_event = threading.Event()
        self.audio_queue = asyncio.Queue()

    def stop(self):
        self._stop_event.set()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put_nowait(indata.tobytes())

    async def sender(self, ws):
        try:
            while not self._stop_event.is_set():
                audio_data = await self.audio_queue.get()
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_data).decode('utf-8')
                }
                await ws.send(json.dumps(payload))
                self.audio_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def receiver(self, ws):
        try:
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
                    if transcript:
                        print(f"English Transcript: {transcript}")
                        self.transcript_queue.put(transcript)
                elif data.get("type") == "error":
                    print(f"An error occurred: {data}")
                    self.status_queue.put(f"Error: {data['error']['message']}")
                    break
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
            self.status_queue.put("Connection Lost")
        except Exception as e:
            print(f"Receiver error: {e}")
            self.status_queue.put(f"Error: {str(e)}")

    async def run_transcription(self):
        uri = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(certifi.where())
        
        try:
            async with websockets.connect(uri, additional_headers=headers, ssl=ssl_context) as ws:
                self.status_queue.put("Connected. Listening...")
                await ws.send(json.dumps({
                    "type": "transcription_session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                        "turn_detection": {"type": "server_vad"}
                    }
                }))
                sender_task = asyncio.create_task(self.sender(ws))
                receiver_task = asyncio.create_task(self.receiver(ws))
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=BLOCK_SIZE, callback=self.audio_callback):
                    while not self._stop_event.is_set():
                        await asyncio.sleep(0.1)
                sender_task.cancel()
                receiver_task.cancel()
                await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
        except Exception as e:
            print(f"Connection failed: {e}")
            self.status_queue.put(f"Connection Failed: {str(e)}")

    def run(self):
        try:
            asyncio.run(self.run_transcription())
        finally:
            self.status_queue.put("Stopped")

class TranslationWorker(threading.Thread):
    def __init__(self, transcript_queue, translation_queue, language_map):
        super().__init__()
        self.transcript_queue = transcript_queue
        self.translation_queue = translation_queue
        self.language_map = language_map
        self._stop_event = threading.Event()
        self.client = openai.OpenAI(api_key=API_KEY)

    def stop(self):
        self._stop_event.set()
        self.transcript_queue.put(None)

    def translate(self, text, lang_name, lang_code):
        try:
            prompt = (
                f"You are an expert translator. Translate the following English text to {lang_name}. "
                f"Provide ONLY the translation, with no additional text, quotes, or explanations.\n\n"
                f"English text: \"{text}\""
            )
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=len(text) * 3
            )
            translation = response.choices[0].message.content.strip()
            self.translation_queue.put((lang_name, translation))
        except Exception as e:
            print(f"Could not translate to {lang_name}: {e}")
            self.translation_queue.put((lang_name, f"[Error: {e}]"))

    def run(self):
        while not self._stop_event.is_set():
            english_text = self.transcript_queue.get()
            if english_text is None:
                break
            # Also put the English text in the queue for display
            self.translation_queue.put(("English", english_text))
            threads = []
            for lang_name, lang_code in self.language_map.items():
                t = threading.Thread(target=self.translate, args=(english_text, lang_name, lang_code))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

# --- GUI Application (ENHANCED) ---

class TranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shoals Live Meeting Translator")
        self.geometry("1600x900")
        self.configure(bg=COLORS['bg'])
        
        # Set minimum window size
        self.minsize(1400, 700)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Queues for inter-thread communication
        self.transcript_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.status_queue = queue.Queue()

        # Worker threads
        self.transcription_worker = None
        self.translation_worker = None

        # Create main UI
        self.create_header_frame()
        self.create_english_frame()
        self.create_translation_columns()
        
        # Start a poller to check queues from threads
        self.after(100, self.poll_queues)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_header_frame(self):
        """Creates the header with logo, title, and start/stop button."""
        # --- Header container ---
        header_frame = tk.Frame(self, bg=COLORS['bg'], height=70)
        header_frame.pack(fill=tk.X, pady=(10, 0))
        header_frame.pack_propagate(False)

        # Three‐column grid for logo, title, controls
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=2)
        header_frame.grid_columnconfigure(2, weight=1)

        # --- Logo (left) ---
        logo_frame = tk.Frame(header_frame, bg=COLORS['bg'])
        logo_frame.grid(row=0, column=0, sticky='w', padx=20)

        logo_path = resource_path('Shoals_Master_Logo_Green-logo-white-text.png')
        try:
            img = Image.open(logo_path)
            img = img.resize((180, 50), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            logo_label = tk.Label(logo_frame, image=self.logo, bg=COLORS['bg'])
        except Exception as e:
            print(f"Couldn’t load logo at {logo_path}: {e}")
            logo_label = tk.Label(
                logo_frame,
                text="SHOALS",
                bg=COLORS['bg'],
                fg=COLORS['accent'],
                font=("Helvetica", 16, "bold")
            )
        logo_label.pack()

        # --- Title (center) ---
        title_frame = tk.Frame(header_frame, bg=COLORS['bg'])
        title_frame.grid(row=0, column=1)
        title_label = tk.Label(
            title_frame,
            text="LIVE MEETING TRANSLATOR",
            bg=COLORS['bg'],
            fg=COLORS['text'],
            font=("Helvetica", 28, "bold")
        )
        title_label.pack()

        # --- Control button (right) ---
        button_frame = tk.Frame(header_frame, bg=COLORS['bg'])
        button_frame.grid(row=0, column=2, sticky='e', padx=20)
        self.toggle_button = tk.Button(
            button_frame,
            text="START LISTENING",
            command=self.toggle_listening,
            font=("Helvetica", 14, "bold"),
            bg=COLORS['success'],
            fg=COLORS['bg'],
            bd=0,
            padx=25,
            pady=12,
            cursor="hand2",
            activebackground='#6ECC5A',
            highlightthickness=0
        )
        self.toggle_button.pack()

    def create_english_frame(self):
        """Creates the English transcription display area."""
        english_container = tk.Frame(self, bg=COLORS['bg'])
        # remove the extra bottom padding here:
        english_container.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        english_label = tk.Label(
            english_container, text="ENGLISH",
            bg=COLORS['bg'], fg=COLORS['accent'],
            font=("Helvetica", 12, "bold")
        )
        english_label.pack(anchor='w', pady=(0, 5))
        
        self.english_text = scrolledtext.ScrolledText(
            english_container,
            wrap=tk.WORD,
            height=6,
            font=("Helvetica", 16, "bold"),
            bg=COLORS['secondary'],
            fg=COLORS['text'],
            relief=tk.FLAT,
            padx=15,
            pady=4,                        # <–– zero out the internal bottom padding
            highlightbackground=COLORS['border'],
            highlightthickness=1
        )
        self.english_text.pack(fill=tk.X)
        self.english_text.configure(state='disabled')
        
        # restyle the scrollbar to match
        self.english_text.vbar.config(
            bg=COLORS['secondary'],
            troughcolor=COLORS['border'],
            activebackground=COLORS['accent'],
            highlightbackground=COLORS['border'],
            relief='flat'
        )

    def create_translation_columns(self):
        """Creates vertical columns for each translation."""
        # Main container with minimal padding
        columns_container = tk.Frame(self, bg=COLORS['bg'])
        columns_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=(0, 20))
        
        # Configure grid for equal columns
        for i in range(4):
            columns_container.grid_columnconfigure(i, weight=1)
        columns_container.grid_rowconfigure(0, weight=1)
        
        self.translation_text_areas = {}
        
        # Subtle accent colors
        lang_accents = {
            "Spanish": COLORS['accent'],
            "Swahili": COLORS['accent'], 
            "Burmese": COLORS['accent'],
            "Pashto": COLORS['accent']
        }
        
        for i, lang_name in enumerate(TARGET_LANGUAGES):
            # Column container
            column_frame = tk.Frame(
                columns_container, bg=COLORS['secondary'],
                highlightbackground=COLORS['border'],
                highlightthickness=1
            )
            column_frame.grid(row=0, column=i, sticky="nsew", padx=5)
            
            # Minimal header
            header = tk.Label(
                column_frame, text=lang_name.upper(),
                bg=COLORS['secondary'], fg=lang_accents[lang_name],
                font=("Helvetica", 12, "bold")
            )
            header.pack(pady=(10, 5))
            
            # Divider line
            divider = tk.Frame(column_frame, bg=COLORS['border'], height=1)
            divider.pack(fill=tk.X, padx=15)
            
            # Text area
            text_area = scrolledtext.ScrolledText(
                column_frame,
                wrap=tk.WORD,
                font=("Helvetica", 22),
                bg=COLORS['secondary'],
                fg=COLORS['text'],
                relief=tk.FLAT,
                padx=15,
                pady=10,
                highlightthickness=0
            )
            text_area.pack(expand=True, fill=tk.BOTH, pady=(5, 10))
            text_area.configure(state='disabled')
            
            # Style the vertical scrollbar
            text_area.vbar.config(
                bg=COLORS['secondary'],
                troughcolor=COLORS['border'],
                activebackground=COLORS['accent'],
                highlightbackground=COLORS['border'],
                relief='flat'
            )
            
            self.translation_text_areas[lang_name] = text_area
        
        # Status bar at bottom
        status_frame = tk.Frame(self, bg=COLORS['primary'], height=30)
        status_frame.pack(fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame, text="Ready",
            bg=COLORS['primary'], fg=COLORS['text_light'],
            font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

    def toggle_listening(self):
        if self.transcription_worker and self.transcription_worker.is_alive():
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        self.toggle_button.config(text="STOP LISTENING", bg=COLORS['danger'],
                                activebackground='#dd3333')
        self.status_label.config(text="Initializing...")

        self.transcription_worker = TranscriptionWorker(self.transcript_queue, self.status_queue)
        self.translation_worker = TranslationWorker(self.transcript_queue, self.translation_queue, TARGET_LANGUAGES)

        self.transcription_worker.start()
        self.translation_worker.start()

    def stop_listening(self):
        self.toggle_button.config(text="START LISTENING", bg=COLORS['success'],
                                activebackground='#6ECC5A')
        self.status_label.config(text="Stopping...")

        if self.transcription_worker:
            self.transcription_worker.stop()
        if self.translation_worker:
            self.translation_worker.stop()
        
        self.after(500, self.cleanup_threads)

    def cleanup_threads(self):
        if self.transcription_worker and self.transcription_worker.is_alive():
            self.transcription_worker.join(timeout=1)
        if self.translation_worker and self.translation_worker.is_alive():
            self.translation_worker.join(timeout=1)
        self.transcription_worker = None
        self.translation_worker = None
        self.status_label.config(text="Ready")

    def poll_queues(self):
        try:
            status_msg = self.status_queue.get_nowait()
            self.status_label.config(text=status_msg)
        except queue.Empty:
            pass

        try:
            lang, text = self.translation_queue.get_nowait()

            if lang == "English":
                # Update English transcription
                self.english_text.configure(state='normal')
                self.english_text.insert(tk.END, text + "\n")
                # force the view all the way to the bottom
                self.english_text.yview_moveto(1.0)
                self.english_text.configure(state='disabled')

            elif lang in self.translation_text_areas:
                # unchanged for the other languages
                text_area = self.translation_text_areas[lang]
                text_area.configure(state='normal')
                text_area.insert(tk.END, text + "\n\n")
                text_area.see(tk.END)
                text_area.configure(state='disabled')

        except queue.Empty:
            pass

        self.after(100, self.poll_queues)

    def on_closing(self):
        if self.transcription_worker and self.transcription_worker.is_alive():
            self.stop_listening()
        self.destroy()

if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()