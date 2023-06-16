import cv2
import pyautogui
from mediapipe.python.solutions import hands

from hand_processor import HandProcessor

cap = cv2.VideoCapture(0)
hands = hands.Hands()

hand_processor = HandProcessor()
screen_width, screen_height = pyautogui.size()

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    hand_processor.process_hands(results)
    if hand_processor.active:
        hand_processor.draw_fingers(image)

        if not hand_processor.calibrated and hand_processor.is_new_click:
            hand_processor.set_calibration(hand_processor.right_index_position)
            continue

        hand_processor.draw_calibration_rectangle(image)

        if hand_processor.calibrated:
            cx, cy = hand_processor.get_position_in_screen(
                (screen_width, screen_height)
            )
            pyautogui.moveTo(cx, cy)

        if hand_processor.calibrated and hand_processor.is_clicking:
            pyautogui.click()

    cv2.imshow("output", image)
    cv2.waitKey(1)
