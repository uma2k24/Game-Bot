import time
import pyautogui
import os
import json
from pynput import mouse, keyboard

# Fare ve klavye olaylarını kaydetmek için listeler
mouse_events = []
keyboard_events = []

def capture_screen(filename):
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)

# Fare olaylarını işleyen işlevler
def on_click(x, y, button, pressed):
    mouse_events.append({
        'type': 'click',
        'x': x,
        'y': y,
        'button': str(button),
        'pressed': pressed,
        'timestamp': time.time()
    })

def on_move(x, y):
    mouse_events.append({
        'type': 'move',
        'x': x,
        'y': y,
        'timestamp': time.time()
    })

def on_scroll(x, y, dx, dy):
    mouse_events.append({
        'type': 'scroll',
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy,
        'timestamp': time.time()
    })

# Klavye olaylarını işleyen işlevler
def on_press(key):
    try:
        keyboard_events.append({
            'type': 'press',
            'key': key.char,
            'timestamp': time.time()
        })
    except AttributeError:
        keyboard_events.append({
            'type': 'press',
            'key': str(key),
            'timestamp': time.time()
        })

def on_release(key):
    try:
        keyboard_events.append({
            'type': 'release',
            'key': key.char,
            'timestamp': time.time()
        })
    except AttributeError:
        keyboard_events.append({
            'type': 'release',
            'key': str(key),
            'timestamp': time.time()
        })

def main():
    save_path = 'Data/Screenshots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Fare ve klavye dinleyicilerini başlat
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    mouse_listener.start()
    keyboard_listener.start()

    try:
        while True:
            timestamp = int(time.time())
            capture_screen(os.path.join(save_path, f'{timestamp}.png'))
            time.sleep(1)  # Her saniye ekran görüntüsü al
    except KeyboardInterrupt:
        # Dinleyicileri durdur
        mouse_listener.stop()
        keyboard_listener.stop()

        # Olayları JSON formatında kaydet
        with open('mouse_events.json', 'w') as mouse_file:
            json.dump(mouse_events, mouse_file, indent=4)

        with open('keyboard_events.json', 'w') as keyboard_file:
            json.dump(keyboard_events, keyboard_file, indent=4)

        print("Olaylar kaydedildi.")

if __name__ == '__main__':
    main()
