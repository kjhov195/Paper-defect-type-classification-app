import cv2

cap = cv2.VideoCapture('http://192.168.200.143:8080/video')

img_counter = 0
while True:
    ret, frame = cap.read()
    resized = cv2.resize(frame, (600, 400))
    cv2.imshow("Frame", resized)

    key = cv2.waitKey(1)

    if key%256 == 27:
        print("Escape hit, closing the app")
        break
    
    elif key%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("screenshot taken")
        img_counter += 1


cap.release()
cv2.destroyAllwindows()