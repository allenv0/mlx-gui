"""
macOS system tray integration for MLX-GUI.
"""

import logging
import subprocess
import threading
import time
import webbrowser
from typing import Optional

import requests
import rumps

from mlx_gui.cli import start as start_server

logger = logging.getLogger(__name__)


class MLXTrayApp(rumps.App):
    """MLX-GUI system tray application for macOS."""
    
    def __init__(self, port: int = 8000, host: str = "127.0.0.1"):
        super().__init__("MLX", icon="🧠", title="MLX")
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.server_thread = None
        self.server_running = False
        
        # Status tracking
        self.system_status = {}
        self.loaded_models_count = 0
        self.memory_usage = "0GB"
        
        # Create menu items
        self.status_item = rumps.MenuItem("Status: Starting...", callback=None)
        self.models_item = rumps.MenuItem("Models: Loading...", callback=None)
        self.memory_item = rumps.MenuItem("Memory: Loading...", callback=None)
        self.separator1 = rumps.MenuItem("---", callback=None)
        self.admin_item = rumps.MenuItem("Open Admin Interface", callback=self.open_admin)
        self.unload_item = rumps.MenuItem("Unload All Models", callback=self.unload_all_models)
        self.separator2 = rumps.MenuItem("---", callback=None)
        self.quit_item = rumps.MenuItem("Quit MLX-GUI", callback=self.quit_app)
        
        # Add items to menu
        self.menu = [
            self.status_item,
            self.models_item, 
            self.memory_item,
            self.separator1,
            self.admin_item,
            self.unload_item,
            self.separator2,
            self.quit_item
        ]
        
        # Start status update timer (every 10 seconds)
        self.status_timer = rumps.Timer(self.update_status, 10)
        self.status_timer.start()
        
    def start_server_background(self):
        """Start the MLX-GUI server in a background thread."""
        try:
            logger.info(f"Starting MLX-GUI server on {self.host}:{self.port}")
            self.server_running = True
            # Start the server (this will block)
            start_server(
                port=self.port,
                host=self.host,
                reload=False,
                workers=1,
                log_level="info",
                database_path=None
            )
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.server_running = False
            self.title = "MLX ❌"
            self.status_item.title = "Status: Server Failed"
    
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    self.title = "MLX ✅"
                    self.update_status()
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        logger.error("Server failed to start within timeout")
        self.title = "MLX ❌"
        self.status_item.title = "Status: Server Timeout"
        return False
    
    def update_status(self, sender=None):
        """Update menu items with current system status."""
        try:
            # Get system status
            response = requests.get(f"{self.base_url}/v1/system/status", timeout=5)
            if response.status_code == 200:
                self.system_status = response.json()
                
                # Update status indicators
                status = self.system_status.get('status', 'unknown')
                self.status_item.title = f"Status: {status.title()}"
                
                # Update models count
                model_manager = self.system_status.get('model_manager', {})
                self.loaded_models_count = model_manager.get('loaded_models_count', 0)
                queue_size = model_manager.get('queue_size', 0)
                
                if queue_size > 0:
                    self.models_item.title = f"Models: {self.loaded_models_count} loaded, {queue_size} queued"
                else:
                    self.models_item.title = f"Models: {self.loaded_models_count} loaded"
                
                # Update memory usage
                total_model_memory = model_manager.get('total_model_memory_gb', 0)
                system_memory = self.system_status.get('system', {}).get('memory', {})
                total_system_memory = system_memory.get('total_gb', 0)
                
                self.memory_usage = f"{total_model_memory:.1f}GB / {total_system_memory:.1f}GB"
                memory_percent = model_manager.get('memory_usage_percent', 0)
                self.memory_item.title = f"Memory: {self.memory_usage} ({memory_percent:.1f}%)"
                
                # Update tray icon status
                if status == 'running':
                    if self.loaded_models_count > 0:
                        self.title = f"MLX ({self.loaded_models_count})"
                    else:
                        self.title = "MLX ✅"
                else:
                    self.title = "MLX ⚠️"
                    
                # Enable/disable unload button
                self.unload_item.set_callback(self.unload_all_models if self.loaded_models_count > 0 else None)
                
            else:
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get status: {e}")
            self.status_item.title = "Status: Server Unreachable"
            self.models_item.title = "Models: Unknown"
            self.memory_item.title = "Memory: Unknown"
            self.title = "MLX ❌"
            self.unload_item.set_callback(None)
    
    def open_admin(self, sender):
        """Open the admin interface in the default browser."""
        admin_url = f"{self.base_url}/admin"
        logger.info(f"Opening admin interface: {admin_url}")
        
        try:
            # Use webbrowser module for better cross-platform support
            webbrowser.open(admin_url)
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            # Fallback to macOS open command
            try:
                subprocess.run(["open", admin_url], check=True)
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to open with system command: {e2}")
                rumps.alert(
                    title="Error",
                    message=f"Could not open admin interface.\nPlease manually visit: {admin_url}",
                    ok="OK"
                )
    
    def unload_all_models(self, sender):
        """Unload all currently loaded models."""
        if self.loaded_models_count == 0:
            return
            
        # Confirm action
        response = rumps.alert(
            title="Unload All Models",
            message=f"Are you sure you want to unload all {self.loaded_models_count} loaded models?",
            ok="Unload All",
            cancel="Cancel"
        )
        
        if response == 1:  # OK clicked
            try:
                # Get list of loaded models
                models_response = requests.get(f"{self.base_url}/v1/manager/status", timeout=10)
                if models_response.status_code == 200:
                    manager_status = models_response.json()
                    loaded_models = manager_status.get('loaded_models', {})
                    
                    unloaded_count = 0
                    failed_count = 0
                    
                    for model_name in loaded_models.keys():
                        try:
                            unload_response = requests.post(
                                f"{self.base_url}/v1/models/{model_name}/unload",
                                timeout=30
                            )
                            if unload_response.status_code == 200:
                                unloaded_count += 1
                                logger.info(f"Unloaded model: {model_name}")
                            else:
                                failed_count += 1
                                logger.error(f"Failed to unload {model_name}: {unload_response.status_code}")
                        except requests.exceptions.RequestException as e:
                            failed_count += 1
                            logger.error(f"Error unloading {model_name}: {e}")
                    
                    # Show result
                    if failed_count == 0:
                        rumps.notification(
                            title="MLX-GUI",
                            subtitle="Models Unloaded", 
                            message=f"Successfully unloaded {unloaded_count} models",
                            sound=False
                        )
                    else:
                        rumps.alert(
                            title="Unload Complete",
                            message=f"Unloaded: {unloaded_count}\nFailed: {failed_count}",
                            ok="OK"
                        )
                    
                    # Update status immediately
                    self.update_status()
                    
                else:
                    raise requests.exceptions.RequestException(f"HTTP {models_response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to unload models: {e}")
                rumps.alert(
                    title="Error",
                    message=f"Failed to unload models: {e}",
                    ok="OK"
                )
    
    def quit_app(self, sender):
        """Quit the application and stop the server."""
        response = rumps.alert(
            title="Quit MLX-GUI",
            message="Are you sure you want to quit MLX-GUI?\nThis will stop the server and unload all models.",
            ok="Quit",
            cancel="Cancel"
        )
        
        if response == 1:  # OK clicked
            logger.info("Shutting down MLX-GUI...")
            
            # Stop status timer
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            
            # Try graceful shutdown via API
            try:
                requests.post(f"{self.base_url}/shutdown", timeout=5)
            except:
                pass
            
            # Force quit
            rumps.quit_application()
    
    def run(self):
        """Start the tray application."""
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self.start_server_background,
            name="mlx_server",
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready (in another thread to not block tray)
        ready_thread = threading.Thread(
            target=self.wait_for_server,
            name="server_ready_check", 
            daemon=True
        )
        ready_thread.start()
        
        # Run the tray app (this blocks)
        try:
            super().run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.server_running = False


def run_tray_app(port: int = 8000, host: str = "127.0.0.1"):
    """Run the MLX-GUI tray application."""
    try:
        app = MLXTrayApp(port=port, host=host)
        app.run()
    except ImportError as e:
        if "rumps" in str(e):
            logger.error("rumps library not installed. Install with: pip install rumps")
            print("Error: rumps library required for tray mode")
            print("Install with: pip install rumps")
            return False
        else:
            raise
    except Exception as e:
        logger.error(f"Tray app error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # For testing
    import sys
    logging.basicConfig(level=logging.INFO)
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_tray_app(port=port)