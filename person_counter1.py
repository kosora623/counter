import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # YOLOv8 モデルのロード
    # 'yolov8n.pt' は nano モデル（最も軽量）。精度を上げるなら 'yolov8s.pt' (small) など
    model = YOLO('yolov8n.pt')

    # カメラの初期化
    cap = cv2.VideoCapture(0)  # 0 は通常、PCのデフォルトカメラ

    if not cap.isOpened():
        print("カメラを開けませんでした。カメラが接続されているか、または他のアプリケーションで使用されていないか確認してください。")
        return

    # 画面の幅と高さを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 通過ラインの設定（例：画面中央の垂直線）
    line_x = frame_width // 2
    
    # カウント初期値
    person_count = 0

    # 各人物の過去の位置を保存する辞書
    # {track_id: [prev_x, prev_y]}
    tracked_persons_prev_pos = {}
    
    # 各人物が一度ラインを通過したかを記録する辞書
    # {track_id: True/False}
    tracked_persons_passed_line = {}
    
    # YOLOv8で検出するクラス（COCOデータセットでは 'person' が0番目のクラス）
    person_class_id = 0 

    print("YOLOv8を使った人数カウントを開始します。'q' キーで終了。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを読み取れませんでした。")
            break

        # YOLOv8による検出とトラッキング
        # persist=True でトラッキングを継続
        # classes=[person_class_id] で 'person' クラスのみを検出
        results = model.track(frame, persist=True, classes=[person_class_id], verbose=False)
        
        # 検出されたオブジェクトの情報を取得
        # results[0].boxes は検出されたバウンディングボックスのリスト
        # results[0].boxes.data はテンソル形式で [x1, y1, x2, y2, track_id, conf, cls] の情報を持つ
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) # バウンディングボックス座標
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) # トラッキングID
            
            current_frame_tracked_ids = set() # 現在のフレームで検出されたIDのセット

            for bbox, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = bbox
                
                # バウンディングボックスの中心座標
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                current_frame_tracked_ids.add(track_id) # このIDが現在のフレームで検出されたことを記録

                # オブジェクトの描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1) # 重心表示

                # 初めて検出されたIDの場合、初期位置を記録
                if track_id not in tracked_persons_prev_pos:
                    tracked_persons_prev_pos[track_id] = [center_x, center_y]
                    tracked_persons_passed_line[track_id] = False # 初期状態ではラインを通過していない
                
                # ライン通過判定
                # 前のフレームの位置と現在のフレームの位置でラインを横切ったか判断
                prev_x = tracked_persons_prev_pos[track_id][0]

                if not tracked_persons_passed_line[track_id]:
                    # ラインを左から右に通過した場合
                    if prev_x < line_x and center_x >= line_x:
                        person_count += 1
                        tracked_persons_passed_line[track_id] = True
                        print(f"ID {track_id} がラインを横切りました。現在のカウント: {person_count}")
                        cv2.putText(frame, "COUNT!", (center_x, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # ラインを右から左に通過した場合
                    elif prev_x > line_x and center_x <= line_x:
                        person_count += 1
                        tracked_persons_passed_line[track_id] = True
                        print(f"ID {track_id} がラインを横切りました。現在のカウント: {person_count}")
                        cv2.putText(frame, "COUNT!", (center_x, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # 一度ラインを通過したオブジェクトが、ラインから十分離れたら、
                    # 再度カウント可能にするためのリセットロジック
                    # 例：画面の端近くまで移動したらリセット
                    reset_threshold = frame_width // 4 # 画面の1/4まで来たらリセット
                    if (center_x < reset_threshold and prev_x < reset_threshold) or \
                       (center_x > frame_width - reset_threshold and prev_x > frame_width - reset_threshold):
                        tracked_persons_passed_line[track_id] = False
                        # print(f"ID {track_id} の通過状態をリセットしました。")

                # 現在の位置を次回のフレームのために保存
                tracked_persons_prev_pos[track_id] = [center_x, center_y]
            
            # 前のフレームで検出されていたが、現在のフレームで検出されなくなったオブジェクトの情報を削除
            # これにより、画面外に出たオブジェクトの情報が残り続けないようにする
            ids_to_remove = [tid for tid in tracked_persons_prev_pos if tid not in current_frame_tracked_ids]
            for tid in ids_to_remove:
                del tracked_persons_prev_pos[tid]
                if tid in tracked_persons_passed_line:
                    del tracked_persons_passed_line[tid]
        else:
            # 現在のフレームで人物が一人も検出されなかった場合、すべてのトラッキング情報をクリア
            # これをしないと、画面から人がいなくなっても古いIDが残り続ける
            tracked_persons_prev_pos.clear()
            tracked_persons_passed_line.clear()


        # 通過ラインの描画
        cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)

        # カウント数の表示
        cv2.putText(frame, f"Count: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 画面に表示
        cv2.imshow("Person Counter with YOLOv8", frame)

        # 'q' キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()