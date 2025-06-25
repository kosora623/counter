import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("カメラを開けませんでした。カメラが接続されているか、または他のアプリケーションで使用されていないか確認してください。")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    line_x = frame_width // 2
    
    person_count = 0

    tracked_persons_prev_pos = {}
    
    # {track_id: True/False}
    tracked_persons_passed_line = {}
    
    person_class_id = 0 

    print("YOLOv8を使った人数カウントを開始します。'q' キーで終了。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを読み取れませんでした。")
            break

        results = model.track(frame, persist=True, classes=[person_class_id], verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            current_frame_tracked_ids = set()

            for bbox, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = bbox
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                current_frame_tracked_ids.add(track_id) 

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1) 

                if track_id not in tracked_persons_prev_pos:
                    tracked_persons_prev_pos[track_id] = [center_x, center_y]
                    tracked_persons_passed_line[track_id] = False 
                
                prev_x = tracked_persons_prev_pos[track_id][0]

                if not tracked_persons_passed_line[track_id]:
                    if prev_x < line_x and center_x >= line_x:
                        person_count += 1
                        tracked_persons_passed_line[track_id] = True
                        print(f"ID {track_id} がラインを横切りました。現在のカウント: {person_count}")
                        cv2.putText(frame, "COUNT!", (center_x, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    elif prev_x > line_x and center_x <= line_x:
                        person_count += 1
                        tracked_persons_passed_line[track_id] = True
                        print(f"ID {track_id} がラインを横切りました。現在のカウント: {person_count}")
                        cv2.putText(frame, "COUNT!", (center_x, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    
                    reset_threshold = frame_width // 4 
                    if (center_x < reset_threshold and prev_x < reset_threshold) or \
                       (center_x > frame_width - reset_threshold and prev_x > frame_width - reset_threshold):
                        tracked_persons_passed_line[track_id] = False

                tracked_persons_prev_pos[track_id] = [center_x, center_y]
            
            ids_to_remove = [tid for tid in tracked_persons_prev_pos if tid not in current_frame_tracked_ids]
            for tid in ids_to_remove:
                del tracked_persons_prev_pos[tid]
                if tid in tracked_persons_passed_line:
                    del tracked_persons_passed_line[tid]
        else:

            tracked_persons_prev_pos.clear()
            tracked_persons_passed_line.clear()

        cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)

        cv2.putText(frame, f"Count: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Person Counter with YOLOv8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()