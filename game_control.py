import time
import pyautogui
import os

def capture_screen(filename):
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)

def main():
    save_path = 'Data/Screenshots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    while True:
        timestamp = int(time.time())
        capture_screen(os.path.join(save_path, f'{timestamp}.png'))
        time.sleep(1)  # Capture every second

if __name__ == '__main__':
    main()
