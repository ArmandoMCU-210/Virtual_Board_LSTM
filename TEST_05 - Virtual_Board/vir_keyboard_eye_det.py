# vir_keyboard_eye_det.py
import cv2
from time import sleep, time
from detecting_eye_blink_module import FaceMeshDetector
from tracking_hand_module import HandDetector
from pygame import mixer
from autocompleter_module import Autocompleter

# Clase para botones
default_button_size = [80, 80]
class Button:
    def __init__(self, pos, text, size=None):
        self.pos = pos
        self.size = size if size is not None else default_button_size.copy()
        self.text = text

    def draw(self, img, color=(0, 0, 0), text_color=(255, 255, 255)):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(img, self.text, (x + 15, y + int(h*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 4)

    def is_hover(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        return bx < x < bx + bw and by < y < by + bh

# Inicializaciones
overlay_window = "Virtual_keyboard"
cv2.namedWindow(overlay_window, cv2.WINDOW_NORMAL)

mixer.init()
voice_click = mixer.Sound('sclick.mp3')  # Asegúrate de tener este archivo

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

eye_detector = FaceMeshDetector(maxFaces=1)
hand_detector = HandDetector(detectionCon=0.8)
autocomp = Autocompleter()



# Layout del teclado (minúsculas)
Keys = [
    list("qwertyuiop"),
    list("asdfghjklñ"),
    list("zxcvbnm,.?")
]

# Crear botones de letras
buttonList = []
for i, row in enumerate(Keys):
    for j, key in enumerate(row):
        x = 100 * j + 150
        y = 100 * i + 150
        buttonList.append(Button([x, y], key))

# Botones de control (entre teclado y texto)
control_keys = ['Tab', 'Space', 'Backspace', 'Save']
control_buttons = []
w_ctrl, h_ctrl, gap = 150, 60, 30
start_x = 150
start_y = 150 + len(Keys) * 100 + 20
for i, key in enumerate(control_keys):
    x = start_x + i * (w_ctrl + gap)
    y = start_y
    control_buttons.append(Button([x, y], key, [w_ctrl, h_ctrl]))

finalText = ''
suggestion = ''
last_click = 0
click_delay = 0.3

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Detección de rostro y parpadeo
    img, faces = eye_detector.findFaceMesh(img, False)
    blink = False
    if faces:
        blink, img = eye_detector.EyeBlinkDetector(img, faces, True)

    # Detección de mano
    img = hand_detector.findHands(img)
    lmList = hand_detector.findPosition(img, draw=False)

    # Área de texto inferior
    cv2.rectangle(img, (200, 500), (1100, 580), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, finalText, (220, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    # Mostrar sugerencia tras el texto
    if finalText:
        suggestion = autocomp.suggest(finalText)
        width = cv2.getTextSize(finalText, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0][0]
        cv2.putText(img, suggestion, (220 + width, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 5)

    # Dibujar botones
    for btn in buttonList:
        btn.draw(img)
    for btn in control_buttons:
        btn.draw(img, color=(50, 50, 50), text_color=(200, 200, 200))

    # Interacción con parpadeo
    if lmList and blink and time() - last_click > click_delay:
        x, y = lmList[8][1], lmList[8][2]
        # Control
        for btn in control_buttons:
            if btn.is_hover(x, y):
                try:
                    voice_click.play()
                except:
                    pass
                if btn.text == 'Tab' and suggestion:
                    finalText += ' ' + suggestion
                    suggestion = ''
                elif btn.text == 'Space':
                    finalText += ' '
                elif btn.text == 'Backspace':
                    finalText = finalText[:-1]
                elif btn.text == 'Save':
                    with open('output.txt', 'a', encoding='utf-8') as f:
                        f.write(finalText + "\n")
                last_click = time()
        # Letras
        for btn in buttonList:
            if btn.is_hover(x, y):
                try:
                    voice_click.play()
                except:
                    pass
                finalText += btn.text
                last_click = time()

    sleep(0.1)
    cv2.imshow(overlay_window, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
