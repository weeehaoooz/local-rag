import subprocess
import os
import signal
import time

def stop_ollama():
    """Finds and kills all processes with 'ollama' in their name."""
    try:
        # Get list of PIDs for processes matching 'ollama'
        # pgrep -f matches the full command line
        result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        
        if not pids or pids == ['']:
            print("No Ollama processes found running.")
            return

        print(f"Found {len(pids)} Ollama processes. Sending SIGTERM...")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except OSError as e:
                print(f"Could not kill PID {pid}: {e}")

        # Wait a bit and check if they're still there. Use SIGKILL if needed.
        time.sleep(1)
        
        result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
        remaining_pids = result.stdout.strip().split('\n')
        
        if remaining_pids and remaining_pids != ['']:
            print(f"Still {len(remaining_pids)} processes remaining. Sending SIGKILL...")
            for pid in remaining_pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Sent SIGKILL to PID {pid}")
                except OSError:
                    pass
            print("Force-stop completed.")
        else:
            print("All Ollama processes stopped peacefully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    stop_ollama()
