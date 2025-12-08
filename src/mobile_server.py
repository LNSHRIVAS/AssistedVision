"""
Mobile Communication Server
Handles WebSocket communication with mobile phone for:
- Sending audio commands to phone
- Receiving gyroscope data from phone
"""
import asyncio
import websockets
import json
import threading
import queue
from collections import deque

class MobileServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.audio_queue = queue.Queue()
        self.gyro_data = {'roll': 0, 'pitch': 0, 'yaw': 0}
        self.gyro_history = deque(maxlen=10)
        self.server_task = None
        self.loop = None
        self.thread = None
        
    def start(self):
        """Start the WebSocket server in a separate thread"""
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"Mobile server starting on ws://{self.host}:{self.port}")
        
    def _run_server(self):
        """Run the asyncio event loop in thread"""
        async def start_and_run():
            async with websockets.serve(self._handle_client, self.host, self.port):
                await asyncio.Future()  # Run forever
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(start_and_run())
        except Exception as e:
            print(f"WebSocket server error: {e}")
        
    async def _handle_client(self, websocket, path):
        """Handle individual client connection"""
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"Mobile client connected from {client_addr}")
        
        try:
            # Send pending audio commands
            asyncio.create_task(self._send_audio_task(websocket))
            
            # Receive gyro data
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'gyro':
                        self.gyro_data = {
                            'roll': data.get('roll', 0),
                            'pitch': data.get('pitch', 0),
                            'yaw': data.get('yaw', 0)
                        }
                        self.gyro_history.append(self.gyro_data.copy())
                except json.JSONDecodeError:
                    print(f"Invalid JSON from client: {message}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Mobile client {client_addr} disconnected")
        finally:
            self.connected_clients.discard(websocket)
            
    async def _send_audio_task(self, websocket):
        """Continuously send audio commands from queue"""
        while websocket in self.connected_clients:
            try:
                if not self.audio_queue.empty():
                    audio_cmd = self.audio_queue.get_nowait()
                    await websocket.send(json.dumps(audio_cmd))
            except Exception as e:
                print(f"Error sending audio command: {e}")
            await asyncio.sleep(0.01)
            
    def send_audio_command(self, text, urgency=0.0, side=None):
        """Queue an audio command to be sent to mobile"""
        command = {
            'type': 'audio',
            'text': text,
            'urgency': urgency,
            'side': side
        }
        self.audio_queue.put(command)
        
    def get_gyro_data(self):
        """Get latest gyroscope data"""
        return self.gyro_data.copy()
    
    def get_gyro_average(self, n=5):
        """Get average of last n gyro readings for stability"""
        if not self.gyro_history:
            return self.gyro_data.copy()
        
        n = min(n, len(self.gyro_history))
        avg = {'roll': 0, 'pitch': 0, 'yaw': 0}
        for data in list(self.gyro_history)[-n:]:
            for key in avg:
                avg[key] += data[key]
        for key in avg:
            avg[key] /= n
        return avg
    
    def is_connected(self):
        """Check if any mobile client is connected"""
        return len(self.connected_clients) > 0
    
    def stop(self):
        """Stop the server"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)
        print("Mobile server stopped")
