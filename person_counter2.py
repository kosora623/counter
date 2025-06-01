import cv2
import numpy as np
from ultralytics import YOLO

def is_point_in_polygon(point, polygon):
    """
    点がポリゴン内部にあるかを判定する（OpenCVのPointPolygonTestを使用）
    point: (x, y) タプル
    polygon: NumPy配列のポリゴン座標 (例: np.array([[x1, y1], [x2, y2], ...]))
    """
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def main():
    model = YOLO('yolov8n.pt') # YOLOv8 モデルのロード

    cap = cv2.VideoCapture(0) # カメラの初期化

    if not cap.isOpened():
        print("カメラを開けませんでした。カメラが接続されているか、または他のアプリケーションで使用されていないか確認してください。")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # カウント対象エリアの定義 (例: 画面下半分の四角形)
    # ポリゴンは NumPy 配列で指定
    # area_polygon = np.array([[0, frame_height // 2], 
    #                          [frame_width, frame_height // 2],
    #                          [frame_width, frame_height], 
    #                          [0, frame_height]], np.int32)

    # カスタムの多角形エリアを定義することも可能 (例: 台形)
    area_polygon = np.array([[frame_width * 0.2, frame_height * 0.4], # 左上
                             [frame_width * 0.8, frame_height * 0.4], # 右上
                             [frame_width * 0.9, frame_height * 0.9], # 右下
                             [frame_width * 0.1, frame_height * 0.9]], np.int32)
    area_polygon = area_polygon.reshape((-1, 1, 2)) # OpenCVの描画形式に変換

    # エリア内の人物を管理するセット
    persons_in_area = set()
    
    person_class_id = 0 # YOLOv8で検出するクラス（'person'）

    print("YOLOv8を使ったエリア内人数カウントを開始します。'q' キーで終了。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを読み取れませんでした。")
            break

        results = model.track(frame, persist=True, classes=[person_class_id], verbose=False)
        
        current_frame_tracked_ids = set() # 現在のフレームで検出されたIDのセット
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for bbox, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                current_frame_tracked_ids.add(track_id)

                # オブジェクトの描画
                color = (0, 255, 0) # デフォルトは緑
                if is_point_in_polygon((center_x, center_y), area_polygon):
                    persons_in_area.add(track_id) # エリア内にいる人物として追加
                    color = (0, 255, 255) # エリア内にいる場合は黄色

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # エリアから出た人物をセットから削除
        # 現在検出されたIDの中にないが、以前エリア内にいたIDを削除対象とする
        ids_to_remove = [tid for tid in persons_in_area if tid not in current_frame_tracked_ids]
        for tid in ids_to_remove:
            persons_in_area.remove(tid)
        
        # エリアの描写
        # cv2.polylines() は閉じた図形ではなく線を描画するため、thickness > 0
        # cv2.fillPoly() は塗りつぶし
        overlay = frame.copy()
        cv2.polylines(overlay, [area_polygon], True, (255, 0, 0), 3) # エリアの枠を青で描画
        # 半透明の塗りつぶしが欲しい場合は以下を使用 (性能に影響する場合あり)
        # cv2.fillPoly(overlay, [area_polygon], (255, 0, 0)) 
        # frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0) # 半透明にする

        # エリア内の人数を表示
        current_area_count = len(persons_in_area)
        cv2.putText(frame, f"In Area: {current_area_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Person Count in Area with YOLOv8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()