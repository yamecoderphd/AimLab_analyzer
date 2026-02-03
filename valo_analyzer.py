import cv2
import numpy as np
import os
import glob

# ✅ 자동 계산된 ROI 적용
ROI_X, ROI_Y, ROI_W, ROI_H = 520, 265, 809, 641
print(f"✅ 자동 계산된 ROI: ({ROI_X}, {ROI_Y}, {ROI_W}, {ROI_H})")

# 🔥 설정값들
base_folder = r'/Volumes/ボリューム/2025_gaze_experiment/sub5/day11'
red_ratio_threshold = 0.45  # ROI 빨강 45%↑
blue_ratio_threshold = 0.10  # 주변 파랑 10%↑
frame_skip = 3
min_file_size_gb = 1.5
DETECTION_BLOCK_SECONDS = 90


def get_unique_filename(folder_path, base_filename, suffix="_red_blue"):
    """중복 파일명 처리"""
    if base_filename.endswith('.png'):
        base_filename = base_filename[:-4]

    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_filename}{suffix}.png"
        else:
            filename = f"{base_filename}{suffix}({counter}).png"

        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            return filename
        counter += 1


def is_roi_red_and_border_blue(frame):
    """ROI 빨강 45%↑ + ROI 주변 20px 진한 파랑 검출"""

    # 1) ROI 빨강 검출
    roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ROI 빨강 마스크
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)

    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)

    lower_red_bright = np.array([0, 60, 120])
    upper_red_bright = np.array([15, 255, 255])
    mask_bright = cv2.inRange(hsv_roi, lower_red_bright, upper_red_bright)

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.bitwise_or(red_mask, mask_bright)

    red_pixels = cv2.countNonZero(red_mask)
    total_roi_pixels = ROI_W * ROI_H
    red_ratio = red_pixels / total_roi_pixels

    # 2) ROI 주변 20px 진한 파랑 검출
    border_margin = 20

    border_top = max(0, ROI_Y - border_margin)
    border_bottom = min(frame.shape[0], ROI_Y + ROI_H + border_margin)
    border_left = max(0, ROI_X - border_margin)
    border_right = min(frame.shape[1], ROI_X + ROI_W + border_margin)

    border_roi = frame[border_top:border_bottom, border_left:border_right]
    hsv_border = cv2.cvtColor(border_roi, cv2.COLOR_BGR2HSV)

    # 첨부 이미지의 진한 남색/네이비 파랑 범위
    lower_blue1 = np.array([100, 100, 20])
    upper_blue1 = np.array([120, 255, 120])

    lower_blue2 = np.array([105, 120, 10])
    upper_blue2 = np.array([115, 255, 80])

    blue_mask1 = cv2.inRange(hsv_border, lower_blue1, upper_blue1)
    blue_mask2 = cv2.inRange(hsv_border, lower_blue2, upper_blue2)
    blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)

    blue_pixels = cv2.countNonZero(blue_mask)
    total_border_pixels = border_roi.shape[0] * border_roi.shape[1]
    blue_ratio = blue_pixels / total_border_pixels

    return red_ratio, blue_ratio, red_mask, blue_mask


def process_video(video_path):
    """단일 영상 처리"""
    # 🔥 영상이 위치한 폴더에 바로 저장! (red_captures 폴더 없이)
    video_folder = os.path.dirname(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 처리 중: {os.path.basename(video_path)} | {total_frames // fps / 60:.1f}분")
    print(f"📁 캡쳐 저장: {video_folder}")  # 영상과 같은 폴더에 저장

    frame_idx = 0
    capture_idx = 1
    max_red_ratio = 0
    max_blue_ratio = 0
    capture_count = 0
    detection_blocked_until = 0

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        if current_time < detection_blocked_until:
            frame_idx += 1
            continue

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        red_ratio, blue_ratio, red_mask, blue_mask = is_roi_red_and_border_blue(frame)

        if red_ratio > max_red_ratio:
            max_red_ratio = red_ratio
        if blue_ratio > max_blue_ratio:
            max_blue_ratio = blue_ratio

        # 🔥 ROI 빨강 45%↑ + 주변 파랑 10%↑ 둘 다 만족!
        if red_ratio > red_ratio_threshold and blue_ratio > blue_ratio_threshold:
            filename = get_unique_filename(video_folder, base_name, f"_red_blue_{int(current_time):06d}s")
            save_path = os.path.join(video_folder, filename)

            cv2.imwrite(save_path, frame)
            print(f"🔴🔵 캡쳐 #{capture_idx}: {filename} "
                  f"(ROI_red={red_ratio:.3f}, border_blue={blue_ratio:.3f}, t={current_time:.1f}s)")

            detection_blocked_until = current_time + DETECTION_BLOCK_SECONDS
            print(
                f"   ⏸️  {DETECTION_BLOCK_SECONDS}초 차단 (~{int(detection_blocked_until // 60):02d}:{int(detection_blocked_until % 60):02d})")

            capture_count += 1
            capture_idx += 1

        if frame_idx % 300 == 0:
            block_status = "차단중" if current_time < detection_blocked_until else "검출중"
            print(f"\r🌈 {frame_idx / total_frames * 100:.1f}% | "
                  f"red={red_ratio:.2f} blue={blue_ratio:.2f} | {block_status}", end="")

        frame_idx += 1

    cap.release()
    print(f"\n✅ 완료 | 캡쳐: {capture_count}개 | 최고 red={max_red_ratio:.3f}, blue={max_blue_ratio:.3f}")
    return capture_count


# 🔥 3GB 이상 영상만 처리
print(f"🔍 {base_folder}에서 3GB↑ 영상만 검색 중...")

video_files = []
small_files = []
video_extensions = ['*.mp4', '*.mkv', '*.avi', '*.mov']
min_file_size_bytes = min_file_size_gb * 1024 ** 3

for root, dirs, files in os.walk(base_folder):
    for ext in video_extensions:
        videos = glob.glob(os.path.join(root, ext))
        for video_path in videos:
            file_size = os.path.getsize(video_path)
            file_size_gb = file_size / 1024 ** 3

            if file_size >= min_file_size_bytes:
                video_files.append(video_path)
                print(f"✅ 포함: {os.path.basename(video_path)} ({file_size_gb:.1f}GB)")
            else:
                small_files.append((os.path.basename(video_path), file_size_gb))

print(f"🎬 3GB↑ 처리대상: {len(video_files)}개 | 3GB미만 제외: {len(small_files)}개")
print("=" * 80)

total_captures = 0
for idx, video_path in enumerate(video_files, 1):
    print(f"\n[{idx}/{len(video_files)}]")
    captures = process_video(video_path)
    total_captures += captures

print(f"\n🎉 3GB↑ 영상 처리 완료! 총 {total_captures}개 캡쳐")
