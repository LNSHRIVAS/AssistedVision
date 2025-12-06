import threading, queue, time, subprocess, sys
class TTSWorker:
    def __init__(self):
        self.q = queue.Queue()
        self.use_powershell = sys.platform == 'win32'  # Use PowerShell on Windows
        
        if self.use_powershell:
            print("[TTS] Using Windows PowerShell SAPI for audio")
            # Test PowerShell TTS
            try:
                subprocess.run([
                    'powershell', '-Command',
                    'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 1; $synth.Volume = 100; $synth.Speak("Audio test. Can you hear me?");'
                ], check=True, capture_output=True, timeout=5)
                print("[TTS] PowerShell audio test complete")
            except Exception as e:
                print(f"[TTS ERROR] PowerShell test failed: {e}")
        else:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('volume', 1.0)
                self.base_rate = self.engine.getProperty('rate')
                self.base_vol = 1.0
                print(f"[TTS] Using pyttsx3 (rate={self.base_rate}, vol={self.base_vol})")
            except Exception as e:
                print(f"[TTS ERROR] Failed to initialize: {e}")
                self.engine = None
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def _run(self):
        while self.running:
            try:
                text, urgency = self.q.get(timeout=0.5)
            except Exception:
                continue
            
            try:
                if self.use_powershell:
                    # Use PowerShell SAPI for Windows
                    rate = int(1 + urgency * 2)  # 1-3 range
                    volume = 100  # Max volume
                    # Escape single quotes in text
                    escaped_text = text.replace("'", "''")
                    ps_command = f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = {rate}; $synth.Volume = {volume}; $synth.Speak('{escaped_text}');"
                    
                    print(f"[TTS] Speaking via PowerShell: '{text[:50]}...' (urgency={urgency:.2f})")
                    subprocess.run(['powershell', '-Command', ps_command], 
                                 check=True, capture_output=True, timeout=10)
                else:
                    # Use pyttsx3 for other platforms
                    if not hasattr(self, 'engine') or self.engine is None:
                        print("[TTS ERROR] No engine available")
                        self.q.task_done()
                        continue
                    
                    rate = int(self.base_rate + 40*urgency)
                    vol = min(1.0, self.base_vol + 0.5*urgency)
                    self.engine.setProperty('rate', rate)
                    self.engine.setProperty('volume', vol)
                    print(f"[TTS] Speaking via pyttsx3: '{text[:50]}...' (urgency={urgency:.2f})")
                    self.engine.say(text)
                    self.engine.runAndWait()
                
                time.sleep(0.01)
            except Exception as e:
                print(f"[TTS ERROR] Failed to speak: {e}")
            finally:
                self.q.task_done()
    def speak(self, text, urgency=0.0, clear_queue=False):
        # Clear old messages if requested (for fresh priority messages)
        if clear_queue:
            cleared = 0
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                    self.q.task_done()
                    cleared += 1
                except:
                    break
            if cleared > 0:
                print(f"[TTS] Cleared {cleared} queued messages")
        print(f"[TTS] Queued: '{text[:50]}...' (urgency={urgency:.2f}, queue_size={self.q.qsize()+1})")
        self.q.put((text, urgency))
    def stop(self):
        self.running = False; self.thread.join(timeout=1)
