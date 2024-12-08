import speech_recognition as sr
import threading
import keyboard
import time

# Initialize the recognizer
r = sr.Recognizer()

# Global flag to stop the loop
stop_flag = False

# This is the microphone source
mic = sr.Microphone()

def record_text():
    while not stop_flag:
        try:
            # Adjusting microphone sensitivity based on ambient noise levels
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.2)  # Allowing more time for noise calibration
                print("Listening...")  # To show that it is actively listening
                audio = r.listen(source, timeout=3, phrase_time_limit=5)  # Increase timeout and limit phrase time

                # Recognize the speech
                Mytext = r.recognize_google(audio, language="en-US")  # Using the more specific language
                output_text(Mytext)

                # Print live transcription to console
                print(f"Live Transcription: {Mytext}")

        except sr.RequestError as e:
            print(f"Couldn't request results; {0}".format(e))
        except sr.UnknownValueError:
            print("Couldn't understand the audio, please try again.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

def output_text(text):
    """ Write the transcribed text into a file """
    with open("output.txt", "a") as f:
        f.write(text + "\n")

def check_for_exit():
    global stop_flag
    while not stop_flag:
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            stop_flag = True
            print("Stopping the transcription process.")
            break
        time.sleep(0.1)  # Avoid busy-waiting

def main():
    # Start the speech recognition in a separate thread
    recognition_thread = threading.Thread(target=record_text)
    recognition_thread.daemon = True  # Daemonize the thread to allow automatic termination
    recognition_thread.start()

    # Start the keyboard detection in a separate thread
    exit_thread = threading.Thread(target=check_for_exit)
    exit_thread.daemon = True  # Daemonize the thread
    exit_thread.start()

    # Wait for both threads to finish
    recognition_thread.join()
    exit_thread.join()

if __name__ == "__main__":
    main()
