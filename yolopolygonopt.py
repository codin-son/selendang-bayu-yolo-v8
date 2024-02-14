import numpy as np
import supervision as sv
import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
import psycopg2
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Traffic monitoring script")
parser.add_argument("--video", type=str, default="vid/15.mp4", help="Path to the video file")
parser.add_argument("--date", type=str, default="18-10-2023", help="Date in MM-DD-YYYY format")
parser.add_argument("--start_time", type=str, default="16:48:23", help="Start time in HH:MM:SS format")
args = parser.parse_args()

# Extract values from command-line arguments
VIDEO = args.video
date = args.date
start_time_str = args.start_time

# VIDEO = "vid/15.mp4"
# date = "18-10-2023"
# start_time_str = "16:48:23"

# Function to convert string time to datetime object
def str_to_time(time_str):
    return datetime.strptime(time_str, "%H:%M:%S")

# Function to convert datetime object to string time
def time_to_str(time_obj):
    return time_obj.strftime("%H:%M:%S")

model = YOLO('bestepo100.pt')
conn = psycopg2.connect(database = "ciq_traffic", 
                        user = "ciq", 
                        host= '127.0.0.1',
                        password = "admin",
                        port = 5432)
cur = conn.cursor()
insert_query ="""
INSERT INTO cars (cars_track_id, cars_dt_taken, lanes_id, cars_current_total, cars_current_total_lane_1, cars_current_total_lane_2, cars_current_total_lane_3)
VALUES (%s, to_timestamp(%s),%s,%s,%s,%s,%s)
"""
# Set the starting time
current_time = str_to_time(start_time_str)
current_time_str=""
frame_counter = 0
cap = cv2.VideoCapture(VIDEO)
colors = sv.ColorPalette.default()
video_info = sv.VideoInfo.from_video_path(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
# round up fps

delay = 1
# Resize image
ret, img = cap.read()
height, width, _ = img.shape

# Initialize zones and annotators
zones = [
    sv.PolygonZone(np.array([[405, 421],[421, 421],[405, 381],[393, 385],[405, 421]]), video_info.resolution_wh),
    sv.PolygonZone(np.array([
[497, 369],[501, 369],[517, 341],[505, 341],[493, 369]
]), video_info.resolution_wh),
    sv.PolygonZone(np.array([
[473, 321],[441, 301],[429, 305],[473, 329],[473, 317]
]), video_info.resolution_wh),
    sv.PolygonZone(np.array([[119, 475],[511, 259],[519, 275],[483, 299],[491, 311],[459, 339],[583, 355],[587, 371],[567, 427],[459, 407],[431, 391],[423, 407],[387, 431],[359, 391],[159, 531],[119, 475]]), video_info.resolution_wh),
    sv.PolygonZone(np.array([[563, 423],[515, 411],[455, 403],[395, 415],[375, 423],[359, 399],[395, 383],[463, 371],[507, 371],[551, 387],[571, 395],[563, 423]]), video_info.resolution_wh),
    sv.PolygonZone(np.array([[583, 383],[531, 371],[503, 363],[447, 359],[403, 375],[367, 399],[351, 379],[423, 343],[475, 335],[531, 347],[587, 355],[583, 383]]), video_info.resolution_wh),
    sv.PolygonZone(np.array([[535, 259],[495, 267],[439, 303],[343, 359],[367, 371],[423, 343],[467, 335],[491, 311],[503, 283],[535, 267],[535, 255]]), video_info.resolution_wh),
]
vertices = np.array([[679, 478],[487, 410],[383, 430],[359, 398],[131, 546],[71, 478],[511, 238],[559, 238],[559, 254],[487, 326],[603, 334],[747, 394],[747, 450],[683, 490],[679, 478]])
zone_annotators = [sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(i), thickness=1, text_thickness=2, text_scale=1) for i, zone in enumerate(zones)]
box_annotators = [sv.BoxAnnotator(color=colors.by_idx(i), thickness=1, text_thickness=1, text_scale=1) for i in range(len(zones))]

prev_counts = [0] * len(zones)
prev_tracker_ids = [[] for _ in range(len(zones))]
total_counts = [0] * len(zones)
total_all_car = 0
total_car_lane_1 = 0
total_car_lane_2 = 0
total_car_lane_3 = 0

while cap.isOpened():
    if frame_counter == fps:
        # Update the current time and reset the frame counter
        current_time += timedelta(seconds=1)
        current_time_str = time_to_str(current_time)
        frame_counter = 0
    ret, img = cap.read()
    if not ret:
        break
    new_width = width // 3
    new_height = height // 3
    
    img = cv2.resize(img, (new_width, new_height))
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    img = cv2.bitwise_and(img, mask)
    for result in model.track(img,persist=True, imgsz=(864,640), verbose=False, device=0, conf=0.01,line_width=1):
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > 0.01]

        for i, (zone, zone_annotator, box_annotator, prev_count, prev_tracker_id) in enumerate(zip(zones, zone_annotators, box_annotators, prev_counts, prev_tracker_ids)):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            if detections_filtered.tracker_id is not None:
                detected_id = detections_filtered.tracker_id.tolist()
                
            else:
                detected_id = []
            if i < 3 :
                img = box_annotator.annotate(scene=img, detections=detections_filtered, skip_label=True)
            img = zone_annotator.annotate(scene=img)

            if (zone.current_count > prev_count and detected_id not in prev_tracker_id) or prev_tracker_id is None:
                total_counts[i] += zone.current_count - prev_count
                if i < 3:
                    # Ensure current_time_str includes a valid time
                    if current_time_str:
                        dt_taken = datetime.strptime(date + " " + current_time_str, "%d-%m-%Y %H:%M:%S")
                    else:
                        dt_taken = datetime.strptime(date, "%d-%m-%Y")
                    dt_taken_timestamp = dt_taken.timestamp()
                    for id in detected_id:
                        data = (id, dt_taken_timestamp, i+1,total_all_car, total_car_lane_1, total_car_lane_2, total_car_lane_3)
                    #     cur.execute(insert_query, data)
                    # conn.commit()
                    # insert db here
            prev_counts[i] = zone.current_count
            if i < len(prev_tracker_id) and len(prev_tracker_id[i]) > 500:
                prev_tracker_id[i] = prev_tracker_id[i][500:]
            if detected_id:
                prev_tracker_ids[i].append(detected_id)
            if i < len(prev_tracker_ids) and prev_tracker_ids[i] and len(prev_tracker_ids[i][0]) > 500:
                prev_tracker_ids[i] = [row[500:] for row in prev_tracker_ids[i]]

    total_all_car = zones[3].current_count
    total_car_lane_1 = zones[4].current_count
    total_car_lane_2 = zones[5].current_count
    total_car_lane_3 = zones[6].current_count

    # Display text
    cv2.rectangle(img, (500,0), (880, 200), (255,255,255), -1)
    for i, total_count in enumerate(total_counts):
        if i < 3:
            cv2.putText(img, f"Total Count Zone {i + 1}: {total_count}", (500, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (58, 134, 255), 2)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (500, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
      
    print(total_car_lane_1)
    print(total_car_lane_2)
    print(total_car_lane_3)

    frame_counter += 1 
    cv2.imshow("result", img)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cur.close()
conn.close()
cv2.destroyAllWindows()
