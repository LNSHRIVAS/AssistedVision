
import os
import sys
import threading
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2

# Add the parent directory to the Python path to allow for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from directguide.navigator import DirectGuideNavigator

class DirectGuideApp(App):
    def build(self):
        # UI Layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Video feed display
        self.image = Image()
        self.layout.add_widget(self.image)
        
        # Status label
        self.status_label = Label(text="Press Start to Begin", size_hint_y=0.1)
        self.layout.add_widget(self.status_label)

        # Control buttons
        button_layout = BoxLayout(size_hint_y=0.1)
        self.start_button = Button(text="Start")
        self.start_button.bind(on_press=self.start_navigation)
        button_layout.add_widget(self.start_button)
        
        self.stop_button = Button(text="Stop", disabled=True)
        self.stop_button.bind(on_press=self.stop_navigation)
        button_layout.add_widget(self.stop_button)
        
        self.layout.add_widget(button_layout)

        # App state
        self.navigator = None
        self.capture = None
        self.nav_thread = None
        self.is_running = False
        
        return self.layout

    def start_navigation(self, instance):
        self.status_label.text = "Initializing..."
        self.start_button.disabled = True
        
        # Initialize camera and navigator on a background thread to avoid UI freeze
        threading.Thread(target=self._initialize_components).start()

    def _initialize_components(self):
        try:
            # Initialize camera
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.status_label.text = "Error: Could not open camera."
                self.start_button.disabled = False
                return

            # Initialize navigator
            self.navigator = DirectGuideNavigator(
                yolo_weights='yolov8n.onnx',
                imgsz=320,
                enable_tts=True,
                use_mobile_detector=True
            )
            
            self.is_running = True
            self.stop_button.disabled = False
            self.status_label.text = "Running..."
            
            # Start the frame processing loop
            Clock.schedule_interval(self.update_frame, 1.0 / 20.0) # Target 20 FPS

        except Exception as e:
            self.status_label.text = f"Initialization failed: {e}"
            self.start_button.disabled = False

    def stop_navigation(self, instance):
        self.is_running = False
        self.stop_button.disabled = True
        self.start_button.disabled = False
        self.status_label.text = "Press Start to Begin"
        
        # Stop the frame processing loop
        Clock.unschedule(self.update_frame)
        
        # Cleanup resources
        if self.navigator:
            self.navigator.cleanup()
        if self.capture:
            self.capture.release()
            
        self.navigator = None
        self.capture = None

    def update_frame(self, dt):
        if not self.is_running or not self.capture:
            return
            
        ret, frame = self.capture.read()
        if not ret:
            return

        try:
            # Process frame in the navigator
            result = self.navigator.process_frame(frame)
            
            # Get the debug overlay
            output_frame = self.navigator.draw_debug_overlay(frame, result)
            
            # Convert the frame to a Kivy texture and display it
            buf = cv2.flip(output_frame, 0).tobytes()
            texture = Texture.create(size=(output_frame.shape[1], output_frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            
        except Exception as e:
            print(f"Error during frame update: {e}")

    def on_stop(self):
        # Ensure cleanup when the app is closed
        self.stop_navigation(None)

if __name__ == "__main__":
    DirectGuideApp().run()
