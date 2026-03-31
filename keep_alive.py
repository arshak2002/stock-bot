from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    """
    Returns a simple response to UptimeRobot pinging 
    the URL to keep the Replit session awake.
    """
    return "Bot is alive and running!"

def run():
    # Run the server quietly on standard web port 8080
    try:
        import os
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"[*] Keep-alive webserver couldn't start (Local run): {e}")

def keep_alive():
    """
    Spawns a background thread to run the Flask web server 
    without blocking the main trading bot pipeline.
    """
    t = Thread(target=run)
    t.start()
