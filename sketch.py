import numpy as np
import cv2
from collections import deque

def setValues():
    print("")

cv2.namedWindow("Detect")
cv2.createTrackbar("U_Hue", "Detect",180, 180, setValues)
cv2.createTrackbar("U_Sat", "Detect",255, 255, setValues)
cv2.createTrackbar("U_Val", "Detect",255, 255, setValues)
cv2.createTrackbar("L_Hue", "Detect",129, 180, setValues)
cv2.createTrackbar("L_Sat", "Detect",141, 255, setValues)
cv2.createTrackbar("L_Val", "Detect",51, 255, setValues)

points = [deque(maxlen = 1000)]

index = 0

kernel = np.ones((5, 5), np.uint8)

color = (0, 0, 255)

paintWindow = np.zeros((480, 640, 3))+255

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    u_hue = cv2.getTrackbarPos("U_Hue","Detect")
    u_sat = cv2.getTrackbarPos("U_Sat","Detect")
    u_val = cv2.getTrackbarPos("U_Val","Detect")
    l_hue = cv2.getTrackbarPos("L_Hue","Detect")
    l_sat = cv2.getTrackbarPos("L_Sat","Detect")
    l_val = cv2.getTrackbarPos("L_Val","Detect")
    upper_hsv = np.array([u_hue, u_sat, u_val])
    lower_hsv = np.array([l_hue, l_sat, l_val])


    frame = cv2.rectangle(frame, (250, 1), (350, 65),(122,122,122), -1)
    
    cv2.putText(frame, "ERASE", (255, 33),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255, 255, 255), 1, cv2.LINE_AA)
    
    Mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations = 1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations = 1)

    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(cnts) > 0:
        
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        if center[1] <= 65:
            if 250 <= center[0] <= 350:
                points = [deque(maxlen = 1000)]
                index=0
                paintWindow[:, :, :] = 255
        else :
            points[index].appendleft(center)
    else:
        points.append(deque(maxlen = 1000))
        index += 1

    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], color, 2)
            cv2.line(paintWindow, points[j][k - 1], points[j][k], color, 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("Mask", Mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
