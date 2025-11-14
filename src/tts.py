import threading, queue, time, pyttsx3
class TTSWorker:
    def __init__(self):
        self.q = queue.Queue(); self.engine = pyttsx3.init()
        self.base_rate = self.engine.getProperty('rate'); self.base_vol = self.engine.getProperty('volume')
        self.running = True; self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
    def _run(self):
        while self.running:
            try:
                text, urgency = self.q.get(timeout=0.5)
            except Exception:
                continue
            rate = int(self.base_rate + 40*urgency); vol = min(1.0, self.base_vol + 0.5*urgency)
            self.engine.setProperty('rate', rate); self.engine.setProperty('volume', vol)
            self.engine.say(text); self.engine.runAndWait(); time.sleep(0.01); self.q.task_done()
    def speak(self, text, urgency=0.0):
        self.q.put((text, urgency))
    def stop(self):
        self.running = False; self.thread.join(timeout=1)
