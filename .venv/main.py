import cv2
import os

model_dir = "C:/Users/vivek/PycharmProjects/FaceDetection/.venv/Scripts"

ageProto = os.path.join(model_dir,"age_deploy.prototxt")
ageModel = os.path.join(model_dir,"age_net.caffemodel")

genderProto = os.path.join(model_dir,"gender_deploy.prototxt")
genderModel = os.path.join(model_dir,"gender_net.caffemodel")


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageNet = cv2.dnn.readNet(ageModel, ageProto)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 5, minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h), (0,255,0), 2)
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label='{},{}'.format(gender, age)
        cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)



    cv2.imshow('Face Detection', frame)

    k=cv2.waitKey(1)

    if(k==ord('q')):
        break

cv2.release()
cv2.destroyAllWindows()
