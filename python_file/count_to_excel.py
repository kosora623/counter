import openpyxl
from datetime import datetime
import os
from ultralytics import YOLO
import cv2
import time

# --- Excelファイル操作関数 ---
def setup_excel(filename="yolo_attendance.xlsx"):
    """
    Excelファイルが存在しない場合、新規作成してヘッダーを書き込む。
    存在する場合は、既存のブックをロードする。
    """
    if not os.path.exists(filename):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Person Count"
        sheet.append(["Timestamp", "Number of People"])
        workbook.save(filename)
    else:
        workbook = openpyxl.load_workbook(filename)
        sheet = workbook.active
    return workbook, sheet

def record_to_excel(sheet, num_people, workbook, filename="yolo_attendance.xlsx"):
    """
    指定されたシートに人数と現在時刻を記録し、ワークブックを保存する。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append([timestamp, num_people])
    workbook.save(filename)
    print(f"Excelに記録しました: 時刻={timestamp}, 人数={num_people}")

# --- YOLOとWebカメラ処理 ---
def run_yolo_and_record(excel_filename="yolo_attendance.xlsx", model_name="yolov8n.pt"):
    """
    YOLOv8モデルを使用してWebカメラから人数を検出し、Excelに記録する。
    """
    # Excelファイルのセットアップ
    workbook, sheet = setup_excel(excel_filename)

    # YOLOv8モデルのロード
    # 'yolov8n.pt' (nano) は最も軽量なモデル。必要に応じて 'yolov8s.pt' (small) などに変更可能。
    model = YOLO(model_name)

    # Webカメラのキャプチャを開始 (0は通常内蔵カメラ)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("YOLOによる人数検出を開始します。'q'キーで終了します。")
    
    # 記録の頻度 (例: 5秒に1回記録)
    record_interval_sec = 5
    last_record_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # YOLOv8で推論を実行
        # 'person'クラスのIDは、YOLOv8のデフォルトのCOCOデータセットでは '0' です。
        # classes=[0] で「人」のみを検出するようにフィルタリングします。
        results = model(frame, stream=True, classes=[0], verbose=False) # verbose=Falseでログを抑制

        person_count = 0
        annotated_frame = frame.copy() # アノテーション用のフレームをコピー

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 検出されたオブジェクトが「人」であるか確認 (classes=[0]でフィルタリング済みだが念のため)
                if int(box.cls[0]) == 0: # クラスIDが0 (person)
                    person_count += 1
                    # 検出された人にバウンディングボックスとラベルを描画
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 検出された人数を画面に表示
        cv2.putText(annotated_frame, f"People: {person_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 処理中のフレームを表示
        cv2.imshow("YOLO Person Detection", annotated_frame)

        # 指定された間隔でExcelに記録
        current_time = time.time()
        if current_time - last_record_time >= record_interval_sec:
            record_to_excel(sheet, person_count, workbook, excel_filename)
            last_record_time = current_time

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    print("処理を終了しました。")

if __name__ == "__main__":
    run_yolo_and_record()