import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def coordinate(image, parameters):
    slope, intercept = parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_lines(image, lines):
    left = []
    right = []

    if(lines is None):
        return

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if(slope < 0):
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)

    left_line = coordinate(image, left_avg)
    right_line = coordinate(image, right_avg)

    return np.array([left_line, right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if(lines is None):
        return line_image

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(100, height), (930, height), (480, 290)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def frameDetector(image):
    lane_image = np.copy(image)
    regioned_image = region_of_interest(canny(lane_image))
    lines = cv2.HoughLinesP(regioned_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
    lines_avg = average_lines(lane_image, lines)
    line_image = display_lines(lane_image, lines_avg)
    return cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

def image_pipeline(folder):
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            cv2.imwrite("./test_images_output/" + filename  + ".jpg", frameDetector(img))

# image_pipeline('test_images')

video = cv2.VideoCapture('./test_videos/3.mp4')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('./test_videos_output/3.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        constructed_frame = frameDetector(frame)
        out.write(constructed_frame)
    else:
        break

video.release()
out.release()
