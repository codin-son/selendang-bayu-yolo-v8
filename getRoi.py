from EasyROI import EasyROI

import cv2
from pprint import pprint

if __name__ == '__main__':
    video_path = 'vid/1.mp4'

    # Initialize cam
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (800,576))
    height, width, _ = frame.shape
    
    new_width = width // 3
    new_height = height // 3
    
    frame = cv2.resize(frame, (new_width, new_height))
    print("Frame size: ", new_width, new_height)
    roi_helper = EasyROI(verbose=True)

    # # DRAW RECTANGULAR ROI
    # rect_roi = roi_helper.draw_rectangle(frame, 1)
    # print("Rectangle Example:")
    # pprint(rect_roi)

    # frame_temp = roi_helper.visualize_roi(frame, rect_roi)
    # cv2.imshow("frame", frame_temp)
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # DRAW LINE ROI
    line_roi = roi_helper.draw_line(frame, 1)
    print("Line Example:")
    pprint(line_roi)

    frame_temp = roi_helper.visualize_roi(frame, line_roi)
    cv2.imshow("frame", frame_temp)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # # DRAW CIRCLE ROI
    # circle_roi = roi_helper.draw_circle(frame, 1)
    # print("Circle Example:")
    # pprint(circle_roi)

    # frame_temp = roi_helper.visualize_roi(frame, circle_roi)
    # cv2.imshow("frame", frame_temp)
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # # DRAW POLYGON ROI
    # polygon_roi = roi_helper.draw_polygon(frame, 1)
    # print("Polygon Example:")
    # pprint(polygon_roi)

    # frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
    # cv2.imshow("frame", frame_temp)
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # # '''
    # cv2.imshow("frame", frame)
    # key = cv2.waitKey(0)
    # if key & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # # '''
