import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# Colors for palette
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
color_names = ['Blue', 'Green', 'Red', 'Black', 'White']
current_color = (255, 0, 0)

# Canvas and smoothing
canvas = None
prev_x, prev_y = 0, 0
points = deque(maxlen=5)

# Draw color picker
def draw_color_bar(frame):
    for i, color in enumerate(colors):
        y1, y2 = 50 + i * 60, 100 + i * 60
        cv2.rectangle(frame, (10, y1), (60, y2), color, -1)
        cv2.putText(frame, color_names[i], (65, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Detect finger states
def fingers_up(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    for i in range(1, 5):  # skip thumb
        tip = hand_landmarks.landmark[tip_ids[i]].y
        pip = hand_landmarks.landmark[tip_ids[i] - 2].y
        fingers.append(tip < pip)
    return fingers  # [Index, Middle, Ring, Pinky]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_color_bar(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = fingers_up(hand_landmarks)

            # Get fingertip and smooth it
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            points.append((x, y))
            avg_x = int(sum(p[0] for p in points) / len(points))
            avg_y = int(sum(p[1] for p in points) / len(points))

            # Gesture: Fist (clear canvas)
            if sum(fingers) == 0:
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0, 0
                continue

            # Gesture: Eraser (Index + Middle)
            if fingers[0] and fingers[1] and not fingers[2]:
                draw_color = (0, 0, 0)
                thickness = 30
            else:
                draw_color = current_color
                thickness = 5

            # Gesture: Color selection
            if avg_x < 60:
                for i in range(len(colors)):
                    y1, y2 = 50 + i * 60, 100 + i * 60
                    if y1 < avg_y < y2:
                        current_color = colors[i]
                        prev_x, prev_y = 0, 0
            else:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = avg_x, avg_y
                cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), draw_color, thickness)
                prev_x, prev_y = avg_x, avg_y

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas with live feed
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("AI Hand Painter", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
