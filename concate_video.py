import cv2
import numpy as np

video_body = cv2.VideoCapture('./predicts/video/body/test_3.mp4')
video_head = cv2.VideoCapture('./predicts/video/head/test_3.mp4')
video_face = cv2.VideoCapture('./predicts/video/face/test_3.mp4')

fwidth = int(video_body.get(cv2.CAP_PROP_FRAME_WIDTH))
fheight = int(video_body.get(cv2.CAP_PROP_FRAME_HEIGHT))
frate = int(video_body.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"h264")
out = cv2.VideoWriter('human-pose-personal-onnx.mp4', fourcc, frate,(fwidth*3, fheight))

while True:
    ret1, frame1 = video_body.read()
    ret2, frame2 = video_head.read()
    ret3, frame3 = video_face.read()

    if not ret1 or not ret2 or not ret3:
        break

    frame2 = cv2.resize(frame2, (fwidth, fheight))
    frame3 = cv2.resize(frame3, (fwidth, fheight))

    canvas = np.zeros((fheight, fwidth*3, 3), dtype=np.uint8)
    canvas[:, :fwidth] = frame1
    canvas[:, fwidth:2*fwidth] = frame2
    canvas[:, 2*fwidth:] = frame3
    out.write(canvas)

video_body.release()
video_head.release()
video_face.release()