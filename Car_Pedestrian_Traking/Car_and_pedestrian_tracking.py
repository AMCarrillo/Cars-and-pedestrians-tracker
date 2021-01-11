import cv2

#Video showing cars and pedestrians
video = cv2.VideoCapture('Car and pedestrian video.mp4')

#Pre-trained cars and pedestrians classifier
car_tracker_file = 'haarcascade_car.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


#Create c and p classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#Run forever until the video ends
while True:

    #Read the current frame
    (read_successul, frame) = video.read()

    #Safe coding
    if read_successul:
        #Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #Detect cars and pedestrians in different scales
    car = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the cars
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    
    #Display the video with the object spotted
    cv2.imshow('C&P detector', frame)

    #Don't autoclose. The number 1 is real time
    key = cv2.waitKey(1)

    #Stop if Q or q is pressed
    if key==81 or key==113:
        break

#Video release
video.release()
