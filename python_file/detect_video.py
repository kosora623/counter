from ultralytics import YOLO
import cv2

# YOLOv8モデルの読み込み
# yolov8n.ptは軽量なモデルです。より高性能なモデル（yolov8m.ptなど）に変更可能です。
model = YOLO('yolov8n.pt')

# 録画映像のパスを指定
# 例: 'your_video.mp4' または カメラの場合 '0'
video_path = "C:\\Program Files\\test.mp4" # ここにご自身の動画ファイルのパスを入力してください
# video_path = 0  # ウェブカメラを使用する場合

# 動画キャプチャオブジェクトの作成
cap = cv2.VideoCapture(video_path)

# 検出結果を保存する動画の設定（オプション）
# 出力ファイル名
output_video_path = 'output_detection_video.mp4'
# フレームレートと解像度を元の動画から取得
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriterオブジェクトの作成
# FourCCはコーデックを指定します。'mp4v'は多くの環境で動作します。
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


if not cap.isOpened():
    print("エラー: ビデオファイルを開けませんでした。")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8で物体検出を実行
        # stream=True で動画ストリームに対して最適化された処理を行います
        results = model(frame, stream=True)

        # 検出結果をフレームに描画
        # results.plot() は検出結果を可視化して新しいフレームを返します
        for r in results:
            annotated_frame = r.plot()

            # 検出されたオブジェクトの情報を表示（オプション）
            # for box in r.boxes:
            #     cls_id = int(box.cls)
            #     conf = float(box.conf)
            #     x1, y1, x2, y2 = map(int, box.xyxy[0])
            #     label = f"{model.names[cls_id]} {conf:.2f}"
            #     print(f"Detected: {label} at [{x1},{y1},{x2},{y2}]")

        # 検出結果を表示
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # 結果を動画ファイルに書き込む
        out.write(annotated_frame)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    out.release() # VideoWriterを解放
    cv2.destroyAllWindows()
    print(f"検出結果が '{output_video_path}' に保存されました。")