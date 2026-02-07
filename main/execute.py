import subprocess
import time

script1 = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\main.py"
script2 = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\read_messages.py"
script3 = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\helper.py"
# Start both scripts in parallel
p1 = subprocess.Popen(["python", script1])
p2 = subprocess.Popen(["python", script2])
p3 = subprocess.Popen(["python", script3])

try:
    # Keep execute.py alive while the two scripts run
    while True:
        # Optional: you can check if subprocesses are still alive
        if p1.poll() is not None:
            print("Script1 exited!")
            break
        if p2.poll() is not None:
            print("Script2 exited!")
            break
        if p3.poll() is not None:
            print("Script2 exited!")
            break
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping subprocesses...")
    p1.terminate()
    p2.terminate()
    p3.terminate()