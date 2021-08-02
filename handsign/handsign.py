import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import signclassifier
from utils import SIGNS, onehot

model = signclassifier().cuda()

try:
    model.load_state_dict(torch.load("saves/best.pt"))
except:
    print("No model saved")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles


def eval_on_frame(data):
    global model

    model = model.eval()

    data = torch.tensor(data).unsqueeze(0).cuda()
    out = model(data)

    # print(SIGNS[str(torch.argmax(out.squeeze(0)).item())])
    return torch.argmax(out.squeeze(0)).item()


def save_frame(data, sign):
    with open("data/trainning.csv", "a") as f:
        pairs = [f"{d[0]:.2f},{d[1]:.2f}" for d in data]
        line = str(sign) + ";" + ";".join(pairs) + "\n"
        f.write(line)


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        res = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmark_style(),
                    drawing_styles.get_default_hand_connection_style())

                res.append([])
                for j, landmark in enumerate(hand_landmarks.landmark):
                    res[-1].append([
                        landmark.x,
                        landmark.y
                        ])

        ids = []
        if res != []:
            for hand in res:
                id = str(eval_on_frame(hand))
                if id in SIGNS:
                    ids.append(SIGNS[id])
                else:
                    print(f"Id: {id} is not in the database")
            line = " | ".join(ids)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (20, 450)
            fontScale = 0.8
            fontColor = (255, 255, 255)
            tickness = 2
            cv2.putText(img=image,
                        text=line,
                        org=bottomLeftCornerOfText,
                        fontFace=font,
                        fontScale=fontScale,
                        color=fontColor,
                        thickness=tickness)

        cv2.imshow('MediaPipe Hands', image)

        if (key := (cv2.waitKey(5) & 0xFF)) != 255:
            if res != []:
                if key >= 48 and key < 59:
                    save_frame(res[0], key-48)
            if key == 27:
                break

cap.release()
