import cv2
import os
from glob import glob
from ultralytics import YOLO

# モデルのロード
print("1. モデルをロード中...")
model = YOLO('yolov8n.pt')

# 処理する動画ファイルのディレクトリとパターン
# 例: videoフォルダ内の 'output_000.mp4' から 'output_032.mp4' までを処理
video_directory = 'video'
video_pattern = os.path.join(video_directory, 'output_*.mp4') # 'video/output_000.mp4' など

# globを使って動画ファイルのリストを取得
video_files = sorted(glob(video_pattern))

if not video_files:
    print(f"エラー: 指定されたパターン '{video_pattern}' に一致する動画ファイルが見つかりませんでした。")
    exit()
else:
    print(f"2. 以下の動画ファイルを処理します: {len(video_files)} 個")
    for vf in video_files:
        print(f"   - {vf}")

# 出力ディレクトリの作成（存在しない場合）
output_dir = 'processed_videos'
os.makedirs(output_dir, exist_ok=True)
print(f"3. 処理結果は '{output_dir}' フォルダに保存されます。")

print("-" * 50) # 区切り線

for video_path in video_files:
    # ファイル名から拡張子を除いた部分を取得し、出力ファイル名に利用
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_video_path_mp4 = os.path.join(output_dir, f'tracked_{base_name}.mp4')
    output_video_path_avi = os.path.join(output_dir, f'tracked_{base_name}.avi')
    output_trace_image_path = os.path.join(output_dir, f'human_traces_{base_name}.png')

    print(f"\n--- 動画ファイル '{video_path}' の処理を開始します ---")
    print(f"  出力先: 動画 -> {output_video_path_mp4} または {output_video_path_avi}, 軌跡画像 -> {output_trace_image_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル {video_path} を開けません。パスが正しいか、ファイルが破損していないか確認してください。")
        continue # 次のファイルへスキップ

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  動画情報: FPS={fps}, Width={width}, Height={height}")

    video_writer_mp4 = None
    video_writer_avi = None

    # MP4コーデックを試す (mp4v)
    try:
        video_writer_mp4 = cv2.VideoWriter(output_video_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not video_writer_mp4.isOpened():
            print(f"  警告: MP4形式 ({output_video_path_mp4}) での出力に失敗しました。別のコーデックを試します。")
            video_writer_mp4 = None # 失敗したらNoneにする
    except Exception as e:
        print(f"  警告: MP4形式 ({output_video_path_mp4}) での出力に例外が発生しました: {e}. 別のコーデックを試します。")
        video_writer_mp4 = None

    # MP4コーデックが失敗した場合、AVIコーデックを試す (XVID)
    if video_writer_mp4 is None:
        try:
            video_writer_avi = cv2.VideoWriter(output_video_path_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
            if not video_writer_avi.isOpened():
                print(f"  エラー: AVI形式 ({output_video_path_avi}) での出力も失敗しました。システムに適切なコーデックがインストールされているか確認してください。")
                cap.release()
                continue # 次のファイルへスキップ
            else:
                print(f"  出力動画ファイル (AVI形式) を作成しました: {output_video_path_avi}")
        except Exception as e:
            print(f"  エラー: AVI形式 ({output_video_path_avi}) での出力に例外が発生しました: {e}. システムに適切なコーデックがインストールされているか確認してください。")
            cap.release()
            continue # 次のファイルへスキップ
    else:
        print(f"  出力動画ファイル (MP4形式) を作成しました: {output_video_path_mp4}")

    # 実際に使うvideo_writerを決定
    video_writer = video_writer_mp4 if video_writer_mp4 is not None else video_writer_avi
    if video_writer is None: # どちらのwriterも初期化できなかった場合
        print("  重大なエラー: 動画の書き込みに利用可能なコーデックが見つかりませんでした。この動画の処理をスキップします。")
        cap.release()
        continue # 次のファイルへスキップ

    trace_image = None
    first_frame = True
    object_paths = {}

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("  動画の終わりに到達しました。またはフレームの読み込みに失敗しました。")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  処理中: フレーム {frame_count}")

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
    print("  動画処理が完了しました。")

    if trace_image is not None:
        cv2.imwrite(output_trace_image_path, trace_image)
        print(f"  人間の軌跡画像は {output_trace_image_path} に保存されました。")
    else:
        print("  警告: 軌跡画像は生成されませんでした。検出された人がいない可能性があります。")

    cv2.destroyAllWindows()

    print(f"  最終的な出力: 追跡結果動画は {output_video_path_mp4 if video_writer_mp4 is not None else output_video_path_avi} に、人間の軌跡画像は {output_trace_image_path} に保存されました。")
    print("-" * 50) # 区切り線
