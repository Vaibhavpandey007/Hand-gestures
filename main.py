import cv2
import mediapipe as mp
import pyautogui
import time

# --------------------
# Setup
# --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# --------------------
# Camera (stable)
# --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open camera on index 0, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ No camera found!")
    exit()

time.sleep(1)     # warm-up

prev_x = prev_y = None

# --------------------
# Main Loop
# --------------------
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Empty frame received, retrying...")
        continue

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_index].classification[0].label
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mid = landmarks.landmark[mp_hands.HandHandLandmark.INDEX_FINGER_PIP]

            # --------------------
            # LEFT HAND → MOUSE
            # --------------------
            if handedness == "Left":
                mcp_x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                mcp_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

                cursor_x = int(mcp_x * screen_width)
                cursor_y = int(mcp_y * screen_height)

                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                if index_tip.y >= index_mid.y:
                    pyautogui.click()

            # --------------------
            # RIGHT HAND → KEYBOARD ARROWS
            # --------------------
            elif handedness == "Right":
                x = int(index_tip.x * screen_width)
                y = int(index_tip.y * screen_height)

                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y

                    if abs(dx) > abs(dy):
                        if dx > 50:
                            pyautogui.press('right')
                        elif dx < -50:
                            pyautogui.press('left')
                    else:
                        if dy > 50:
                            pyautogui.press('down')
                        elif dy < -50:
                            pyautogui.press('up')

                prev_x, prev_y = x, y

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
