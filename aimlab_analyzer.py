import cv2
import numpy as np
import os
import glob

# 설정값
base_folder = r'/Volumes/ボリューム/2025_gaze_experiment/sub5/day11'
color_change_threshold = 30.0
frame_skip = 3
record_start_sec = 60
block_seconds = 40
max_file_size_gb = 2.0


def get_unique_filename(folder_path, base_filename):
    """중복 파일명 처리: 영상명 → 영상명_aimlab → 영상명_aimlab(1)"""
    if base_filename.endswith('.csv'):
        base_filename = base_filename[:-4]  # .csv 제거

    # 1. 원본 이름 시도
    csv_filename = base_filename + '.csv'
    csv_path = os.path.join(folder_path, csv_filename)

    if not os.path.exists(csv_path):
        return csv_filename

    # 2. 영상명_aimlab.csv 시도
    csv_filename_aimlab = f"{base_filename}_aimlab.csv"
    csv_path_aimlab = os.path.join(folder_path, csv_filename_aimlab)

    if not os.path.exists(csv_path_aimlab):
        return csv_filename_aimlab

    # 3. 영상명_aimlab(1).csv, (2).csv... 순차 증가
    counter = 1
    while True:
        csv_filename_aimlab_num = f"{base_filename}_aimlab({counter}).csv"
        csv_path_aimlab_num = os.path.join(folder_path, csv_filename_aimlab_num)
        if not os.path.exists(csv_path_aimlab_num):
            return csv_filename_aimlab_num
        counter += 1


def process_video(video_path):
    """단일 영상 처리 함수"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 처리 중: {os.path.basename(video_path)} | {total_frames // fps / 60:.1f}분")

    color_history_h = []
    color_history_s = []
    trial_events = []
    sec_counter = 0
    max_std_per_sec = {}
    processed_frames = 0
    detection_blocked_until = 0
    trial_num = 1

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        current_sec = int(current_time)

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        processed_frames += 1
        if current_time < detection_blocked_until:
            frame_idx += 1
            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        current_h_std = np.std(h)
        current_s_std = np.std(s)

        color_history_h.append(current_h_std)
        color_history_s.append(current_s_std)
        if len(color_history_h) > 15:
            color_history_h.pop(0)
            color_history_s.pop(0)

        if current_sec >= record_start_sec and current_sec != sec_counter:
            if sec_counter in max_std_per_sec:
                print(f"\n🌈 {sec_counter:02d}초: 최대변화 {max_std_per_sec[sec_counter]:.1f}")
            sec_counter = current_sec

        max_std_per_sec[current_sec] = max(
            max_std_per_sec.get(current_sec, 0), current_h_std + current_s_std
        )

        if processed_frames % 20 == 0:
            print(f"\r🌈 {frame_idx / total_frames * 100:.1f}%({current_time:.0f}s) | "
                  f"색변화: {current_h_std + current_s_std:.1f}", end="")

        if current_time >= record_start_sec and len(color_history_h) >= 5:
            prev_h_avg = np.mean(color_history_h[:-5])
            prev_s_avg = np.mean(color_history_s[:-5])
            color_change = (current_h_std + current_s_std) - (prev_h_avg + prev_s_avg)

            if color_change > color_change_threshold:
                start_time = current_time - 60
                end_time = current_time
                trial_events.append([trial_num, int(start_time), int(end_time)])

                print(f"\n🌈 색변화↑: {int(current_time // 60):02d}:{int(current_time % 60):02d} "
                      f"변화량: {color_change:.1f}")

                detection_blocked_until = current_time + block_seconds
                print(
                    f"   ⏸️ 40초 블록 (~{int(detection_blocked_until // 60):02d}:{int(detection_blocked_until % 60):02d})")
                trial_num += 1

        frame_idx += 1

    cap.release()
    return trial_events


# 🔥 하위 폴더 탐색 + 3GB 이상 스킵
print(f"🔍 {base_folder}에서 재귀 검색 중... (3GB↑ 스킵)")
video_files = []
skipped_files = []
video_extensions = ['*.mp4', '*.mkv', '*.avi', '*.mov']
max_file_size_bytes = max_file_size_gb * 1024 * 1024 * 1024

for root, dirs, files in os.walk(base_folder):
    for ext in video_extensions:
        videos = glob.glob(os.path.join(root, ext))
        for video_path in videos:
            file_size_bytes = os.path.getsize(video_path)
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

            if file_size_bytes > max_file_size_bytes:
                skipped_files.append((video_path, file_size_gb))
                print(f"⏭️  스킵: {os.path.basename(video_path)} ({file_size_gb:.1f}GB)")
                continue

            video_files.append((root, video_path))

print(f"🎬 처리대상 영상: {len(video_files)}개 | 스킵: {len(skipped_files)}개")
print("-" * 80)

# 🔥 모든 영상 처리
processed_count = 0
for video_idx, (video_folder, video_path) in enumerate(video_files, 1):
    print(f"\n{'=' * 80}")
    print(f"[{video_idx}/{len(video_files)}] {os.path.basename(video_path)}")
    print(f"📁 폴더: {os.path.basename(video_folder)}")
    print("-" * 80)

    trial_events = process_video(video_path)
    processed_count += 1

    # 🔥 중복 파일명 처리 함수 사용
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = get_unique_filename(video_folder, video_base_name)
    csv_path = os.path.join(video_folder, csv_filename)

    print(f"\n{'=' * 50}")
    print(f"✅ [{csv_filename[:-4]}] 결과 ({len(trial_events)}개)")
    for event in trial_events:
        trial_num, start_sec, end_sec = event
        print(f"  {trial_num}. {start_sec}초 → {end_sec}초")

    # CSV 저장
    with open(csv_path, 'w') as f:
        f.write('Trial,시작(초),끝(초)\n')
        for event in trial_events:
            trial_num, start_sec, end_sec = event
            f.write(f'{trial_num},{start_sec},{end_sec}\n')

    print(f"💾 저장: {csv_path}")

print(f"\n🎉 모든 영상 처리 완료! ({processed_count}/{len(video_files)}개)")
if skipped_files:
    print(f"\n⏭️ 스킵된 대용량 영상:")
    for video_path, size_gb in skipped_files[:5]:
        print(f"  - {os.path.basename(video_path)} ({size_gb:.1f}GB)")
