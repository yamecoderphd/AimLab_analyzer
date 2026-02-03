import cv2
import numpy as np

# 비디오 열기
cap = cv2.VideoCapture(r'/Volumes/ボリューム/2025_gaze_experiment/sub2/day10/2026-01-29 15-02-19.mp4')

# FPS 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps  # 총 길이(초)

# 10분 23초 = 623초
target_time = 10*60  # 623초

print(f"📹 총 길이: {total_duration:.1f}초")
print(f"🎯 목표 시간: {target_time}초 ({target_time//60:02d}:{target_time%60:02d})")

if target_time > total_duration:
    print("❌ 목표 시간이 영상 길이를 초과합니다!")
    cap.release()
    exit()

# 목표 시간으로 프레임 이동 (밀리초 단위)
cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)

# 해당 시점 프레임 읽기
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 프레임을 읽을 수 없습니다!")
    exit()

print(f"✅ {target_time//60:02d}:{target_time%60:02d} 시점 프레임 로드 완료")
print(f"   프레임 크기: {frame.shape}")

# 🔥 마우스로 드래그해서 ROI 선택
roi = cv2.selectROI("ROI 선택 (Enter:확인, ESC:취소)", frame, False)
cv2.destroyAllWindows()

if len(roi) == 4:
    roi_x, roi_y, roi_w, roi_h = roi
    print(f"✅ 자동 계산된 ROI: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
    print(f"코드에 넣을 값: roi_x, roi_y, roi_w, roi_h = {roi}")
else:
    print("❌ ROI 선택 취소")
