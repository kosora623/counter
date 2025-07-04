from ultralytics import YOLO
import cv2

# モデルのロード
print("1. モデルをロード中...")
model = YOLO('yolov8n.pt')

# 動画ファイルのパス
video_path = 'C:\\test.mp4' # 実際の動画ファイルパスに置き換えてください
print(f"2. 動画ファイルパス: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"エラー: 動画ファイル {video_path} を開けません。パスが正しいか、ファイルが破損していないか確認してください。")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"3. 動画情報: FPS={fps}, Width={width}, Height={height}")

# 結果の動画と軌跡画像の保存設定
output_video_path_mp4 = 'tracked_video.mp4'
output_video_path_avi = 'tracked_video.avi' # AVI形式も試す
output_trace_image_path = 'human_traces.png'

video_writer_mp4 = None
video_writer_avi = None

# MP4コーデックを試す (mp4v)
try:
    video_writer_mp4 = cv2.VideoWriter(output_video_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not video_writer_mp4.isOpened():
        print(f"警告: MP4形式 ({output_video_path_mp4}) での出力に失敗しました。別のコーデックを試します。")
        video_writer_mp4 = None # 失敗したらNoneにする
except Exception as e:
    print(f"警告: MP4形式 ({output_video_path_mp4}) での出力に例外が発生しました: {e}. 別のコーデックを試します。")
    video_writer_mp4 = None

# MP4コーデックが失敗した場合、AVIコーデックを試す (XVID)
if video_writer_mp4 is None:
    try:
        video_writer_avi = cv2.VideoWriter(output_video_path_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        if not video_writer_avi.isOpened():
            print(f"エラー: AVI形式 ({output_video_path_avi}) での出力も失敗しました。システムに適切なコーデックがインストールされているか確認してください。")
            exit()
        else:
            print(f"4. 出力動画ファイル (AVI形式) を作成しました: {output_video_path_avi}")
    except Exception as e:
        print(f"エラー: AVI形式 ({output_video_path_avi}) での出力に例外が発生しました: {e}. システムに適切なコーデックがインストールされているか確認してください。")
        exit()
else:
    print(f"4. 出力動画ファイル (MP4形式) を作成しました: {output_video_path_mp4}")


# 実際に使うvideo_writerを決定
video_writer = video_writer_mp4 if video_writer_mp4 is not None else video_writer_avi
if video_writer is None: # どちらのwriterも初期化できなかった場合
    print("重大なエラー: 動画の書き込みに利用可能なコーデックが見つかりませんでした。")
    exit()


trace_image = None
first_frame = True
object_paths = {}

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("5. 動画の終わりに到達しました。またはフレームの読み込みに失敗しました。")
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"6. 処理中: フレーム {frame_count}")

    # 推論の実行 (person クラスのみ)
    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results and results[0].boxes:
        if first_frame:
            trace_image = frame.copy()
            first_frame = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist()[:4])
            track_id = box.id.item() if box.id is not None else None

            if track_id is not None:
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if track_id not in object_paths:
                    object_paths.update({track_id: [center]})
                else:
                    object_paths.get(track_id).append(center)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        temp_trace_image = trace_image.copy()
        for path in object_paths.values():
            for i in range(len(path) - 1):
                cv2.line(temp_trace_image, path[i], path[i + 1], (255, 0, 0), 3)
        trace_image = temp_trace_image

    video_writer.write(frame)

# リソースの解放
cap.release()
video_writer.release()
print("7. 動画処理が完了しました。")

if trace_image is not None:
    cv2.imwrite(output_trace_image_path, trace_image)
    print(f"人間の軌跡画像は {output_trace_image_path} に保存されました。")
else:
    print("警告: 軌跡画像は生成されませんでした。検出された人がいない可能性があります。")

cv2.destroyAllWindows()

print(f"最終的な出力: 追跡結果動画は {output_video_path_mp4 if video_writer_mp4 is not None else output_video_path_avi} に、人間の軌跡画像は {output_trace_image_path} に保存されました。")