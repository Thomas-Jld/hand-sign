import cv2 as cv
import mediapipe as mp
import numpy as np
import onnxruntime

from utils import SIGNS, onehot, CameraVideoReader, IntelVideoReader

model = onnxruntime.InferenceSession("saves/handsign.onnx")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def eval_on_frame(data):
    ort_inputs = {model.get_inputs()[0].name: np.array(data, dtype=np.float32)}
    out = model.run(None, ort_inputs)[-1]
    return float(np.max(out)), np.argmax(out)


def save_frame(data, sign):
    with open("data/trainning.csv", "a") as f:
        pairs = [f"{d[0]:.2f},{d[1]:.2f}" for d in data]
        line = str(sign) + ";" + ";".join(pairs) + "\n"
        f.write(line)


# cap = CameraVideoReader()
cap = IntelVideoReader()

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while 1:
        image, _ = cap.next_frame()

        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        res = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                res.append([])
                for j, landmark in enumerate(hand_landmarks.landmark):
                    res[-1].append([
                        landmark.x,
                        landmark.y
                        ])

        ids = []
        if res != []:
            for hand in res:
                per, id = eval_on_frame(hand)
                id = str(id)
                if id in SIGNS:
                    ids.append(SIGNS[id] + " " + str(per))
                else:
                    print(f"Id: {id} is not in the database")
            line = " | ".join(ids)

            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (20, 450)
            fontScale = 0.8
            fontColor = (255, 255, 255)
            tickness = 2
            cv.putText(
                img=image,
                text=line,
                org=bottomLeftCornerOfText,
                fontFace=font,
                fontScale=fontScale,
                color=fontColor,
                thickness=tickness
            )

        cv.imshow('MediaPipe Hands', image)

        key = cv.waitKey(5)
        if key != -1:
            if res != []:
                save_frame(res[0], 0)
            if key == 27:
                break
