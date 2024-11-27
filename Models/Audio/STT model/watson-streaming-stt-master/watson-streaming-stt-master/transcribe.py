#!/usr/bin/env python
import argparse
import base64
import json
import time
import pyaudio
import websocket
from websocket._abnf import ABNF

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FINALS = []
LAST = None


def debug_ws(ws):
    """Print WebSocket debug information."""
    if ws and ws.sock:
        print(f"WebSocket connected: {ws.sock.connected}")
    else:
        print("WebSocket not connected or invalid.")


def stop_transcription(ws):
    """Send stop action to WebSocket and close connection."""
    if ws and ws.sock and ws.sock.connected:
        try:
            data = {"action": "stop"}
            ws.send(json.dumps(data).encode('utf8'))
            ws.close()
        except Exception as e:
            print(f"Error closing WebSocket: {e}")
    else:
        print("WebSocket already closed or not connected.")


def read_audio(ws, timeout):
    """Read audio and send it to the WebSocket for a fixed duration."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if ws and ws.sock and ws.sock.connected:
                ws.send(data, ABNF.OPCODE_BINARY)
            else:
                print("WebSocket disconnected during recording.")
                break
    except Exception as e:
        print(f"Audio streaming error: {e}")
    finally:
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        stop_transcription(ws)


def on_message(ws, msg):
    """Handle incoming messages and display interim and final results."""
    global LAST
    data = json.loads(msg)
    if "results" in data and data["results"]:
        if data["results"][0]["final"]:
            FINALS.append(data)
            LAST = None
        else:
            LAST = data
        transcript = data["results"][0]["alternatives"][0]["transcript"]
        print(f"Transcript: {transcript}")
    elif "warnings" in data:
        print(f"Warning: {data['warnings']}")
    elif "state" in data:
        print(f"State: {data['state']}")


def on_error(ws, error):
    """Handle errors."""
    print(f"Error: {error}")


def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket closure."""
    global LAST
    if LAST:
        FINALS.append(LAST)
    transcript = "".join([x['results'][0]['alternatives'][0]['transcript'] for x in FINALS])
    print("\nFinal Transcript:")
    print(transcript)
    print(f"WebSocket closed with status: {close_status_code}, message: {close_msg}")


def on_open(ws):
    """Send start message and start reading audio."""
    print("WebSocket connection opened.")
    debug_ws(ws)
    if not ws.sock or not ws.sock.connected:
        print("WebSocket connection failed. Exiting.")
        return
    data = {
        "action": "start",
        "content-type": f"audio/l16;rate={RATE}",
        "interim_results": True,
        "word_confidence": False,
        "timestamps": False,
        "max_alternatives": 1
    }
    try:
        ws.send(json.dumps(data).encode('utf8'))
        read_audio(ws, ws.args.timeout)
    except Exception as e:
        print(f"Error during WebSocket operation: {e}")


def get_url():
    """Return the WebSocket URL."""
    return "wss://api.us-south.speech-to-text.watson.cloud.ibm.com/v1/recognize"


def get_auth():
    """Return the authentication credentials."""
    return ("apikey", "J9oj2dzktoItUrnPXDlh2zhuaiDQXUfzzN6t9KdouqP6")  # Replace with your actual API key


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Transcribe speech using Watson Speech-to-Text in real time.')
    parser.add_argument('-t', '--timeout', type=int, default=5, help='Recording duration in seconds.')
    return parser.parse_args()


def main():
    """Main function to start the WebSocket connection."""
    headers = {}
    userpass = ":".join(get_auth())
    headers["Authorization"] = "Basic " + base64.b64encode(userpass.encode()).decode()
    url = get_url()

    websocket.enableTrace(True)  # Enable WebSocket tracing for debugging
    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.args = parse_args()
    ws.on_open = on_open
    ws.run_forever()


if __name__ == "__main__":
    main()
