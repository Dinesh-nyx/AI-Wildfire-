"""
Simple HTTP Server for Wildfire Dashboard
Serves the HTML dashboard and CSV data
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the HTTP server and open browser"""
    
    # Change to the directory containing the files
    os.chdir(Path(__file__).parent)
    
    # Create server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 80)
        print("🔥 WILDFIRE DASHBOARD SERVER")
        print("=" * 80)
        print(f"\n✅ Server running at: http://localhost:{PORT}")
        print(f"📊 Dashboard URL: http://localhost:{PORT}/wildfire_dashboard.html")
        print("\n🌐 Opening dashboard in your default browser...")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80)
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}/wildfire_dashboard.html')
        
        # Serve forever
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
            print("=" * 80)

if __name__ == "__main__":
    start_server()
