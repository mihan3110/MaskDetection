import cv2
import imutils
import numpy
from mtcnn.mtcnn import MTCNN

from main import *


def find_face(path):
    detector = MTCNN()
    image = cv2.imread(path)

    if image.shape[0] < image.shape[1]:
        image = imutils.resize(image, height=1000)
    else:
        image = imutils.resize(image, width=1000)

    image_size = numpy.asarray(image.shape)[0:2]
    faces_boxes = detector.detect_faces(image)
    image_detected = image.copy()

    if faces_boxes:
        face_n = 0
        for face_box in faces_boxes:
            face_n += 1
            x, y, w, h = face_box['box']
            d = h - w
            w = w + d
            x = numpy.maximum(x - round(d / 2), 0)
            x1 = numpy.maximum(x - round(w / 4), 0)
            y1 = numpy.maximum(y - round(h / 4), 0)
            x2 = numpy.minimum(x + w + round(w / 4), image_size[1])
            y2 = numpy.minimum(y + h + round(h / 4), image_size[0])
            cropped = image_detected[y1:y2, x1:x2, :]
            face_image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
            face_file_name = 'face_' + str(face_n) + '.jpg'

            if face_box['confidence'] > 0.9:
                cv2.imwrite('dataset/test/' + face_file_name, face_image)

    if os.listdir('dataset/test'):
        predict(list(map(lambda x: f'dataset/test/{x}', os.listdir('dataset/test'))))
    else:
        print("Директория пуста - лицо не обнаружено")
