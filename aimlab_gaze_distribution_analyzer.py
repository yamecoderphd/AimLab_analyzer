import cv2
import pandas as pd
import os
import numpy as np

# ===== 실제 사용 시 여기만 수정하세요 =====
FRAME_DATA_PATH = r'/Volumes/ボリューム/2025_gaze_experiment/sub3/day8/sub3_day8_pre.csv'
VIDEO_PATH = r'/Volumes/ボリューム/2025_gaze_experiment/sub3/day8/2025-12-03 14-24-06.mp4'
TRIAL_PATH = r'/Volumes/ボリューム/2025_gaze_experiment/sub3/day8/2025-12-03 14-24-06.csv'
SAVE_AS = None


# ========================================

def process_frame_data(frame_path, video_path, trial_path, output_path=None):
    print("🚀 시작!")

    # 1. 동영상 FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"FPS: {fps}")

    # 2. 프레임 데이터 (안전 처리)
    print(f"\n📊 Frame CSV 로드: {frame_path}")
    df_frame = pd.read_csv(frame_path)
    print(f"📊 Frame shape: {df_frame.shape}")
    print(f"📊 Frame 컬럼: {list(df_frame.columns)}")

    # 🔥 수정: x좌표=B열(인덱스1), y좌표=C열(인덱스2), 프레임ID=A열(인덱스0)
    if len(df_frame.columns) < 3:
        print(f"❌ Frame CSV 컬럼 부족: {len(df_frame.columns)}개 (최소 3개 필요)")
        return

    col_frame_id = df_frame.columns[0]  # A열: 프레임 ID
    col_x = df_frame.columns[1]  # B열: x좌표
    col_y = df_frame.columns[2]  # C열: y좌표
    print(f"프레임ID: '{col_frame_id}', x좌표: '{col_x}', y좌표: '{col_y}'")

    # 3. F열 시간 계산 (A열 프레임ID → 초 단위)
    df_frame['F'] = df_frame[col_frame_id].apply(
        lambda x: round(int(float(x)) / fps, 3) if pd.notna(x) and str(x).strip() != '' else None
    )

    # 4. G열 공백 초기화
    df_frame['G'] = ""

    # 5. Trial CSV 파싱 (인코딩 에러 안전 처리)
    print(f"\n🔍 Trial CSV 로드: {trial_path}")
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']

    df_trial = None
    for encoding in encodings:
        try:
            df_trial = pd.read_csv(trial_path, encoding=encoding)
            print(f"✅ Trial 로드 성공: encoding={encoding}")
            print(f"📊 Trial shape: {df_trial.shape}")
            print(f"📊 Trial 컬럼: {list(df_trial.columns)}")
            break
        except UnicodeDecodeError:
            print(f"⚠️  {encoding} 실패, 다음 시도...")
            continue
        except Exception as e:
            print(f"❌ {encoding} 에러: {e}")
            continue

    if df_trial is None:
        print("❌ Trial CSV 모든 인코딩 실패")
        return

    # 6. Trial 시간들 + start/end 행 위치 기록
    trial_ranges = []
    for i in range(min(5, len(df_trial))):
        try:
            start_time = int(float(df_trial.iloc[i, 1]))  # B열
            end_time = int(float(df_trial.iloc[i, 2]))  # C열

            # start 행 찾기
            mask_start = (
                    df_frame['F'].notna() &
                    df_frame['F'].astype(float).apply(lambda x: int(x) == start_time)
            )
            start_rows = df_frame[mask_start].index
            start_row = start_rows[0] if len(start_rows) > 0 else None

            # end 행 찾기
            mask_end = (
                    df_frame['F'].notna() &
                    df_frame['F'].astype(float).apply(lambda x: int(x) == end_time)
            )
            end_rows = df_frame[mask_end].index
            end_row = end_rows[0] if len(end_rows) > 0 else None

            if start_row is not None and end_row is not None:
                trial_ranges.append((start_row, end_row))
                df_frame.loc[start_row, 'G'] = f"Trial{i + 1} start"
                df_frame.loc[end_row, 'G'] = f"Trial{i + 1} end"
                print(f"Trial{i + 1}: {start_time}s(행{start_row}) ~ {end_time}s(행{end_row})")
            else:
                print(f"⚠️  Trial{i + 1} 시간대 데이터 없음")
        except Exception as e:
            print(f"❌ Trial{i + 1} 처리 에러: {e}")
            continue

    # 7. Trial별 B/C열 통계 (start_row ~ end_row 구간)
    print("\n📊 Trial별 통계 계산 (행 번호 기준)...")
    stats_results = []

    for i, (start_row, end_row) in enumerate(trial_ranges):
        trial_num = i + 1

        try:
            # start_row 부터 end_row 까지 데이터 (포함)
            trial_data = df_frame.iloc[start_row:end_row + 1]

            # NaN 제외하고 계산 (B열=x, C열=y)
            x_data = trial_data[col_x].dropna()
            y_data = trial_data[col_y].dropna()

            if len(x_data) > 0 and len(y_data) > 0:
                x_mean = x_data.mean()
                y_mean = y_data.mean()
                x_std = x_data.std()
                y_std = y_data.std()

                stats_results.append({
                    'trial': trial_num,
                    'x_cor_aver': round(x_mean, 1),
                    'y_cor_aver': round(y_mean, 1),
                    'x_sd': round(x_std, 1),
                    'y_sd': round(y_std, 1),
                    'row_count': len(trial_data)
                })
                print(f"Trial{trial_num}: 행{start_row}~{end_row} ({len(trial_data)}행)")
                print(f"  x_avg={x_mean:.1f}, y_avg={y_mean:.1f}, x_sd={x_std:.1f}, y_sd={y_std:.1f}")
            else:
                print(f"⚠️  Trial{trial_num}: 유효 데이터 없음")
        except Exception as e:
            print(f"❌ Trial{trial_num} 통계 계산 에러: {e}")
            continue

    # 8. 통계 저장
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_file = os.path.splitext(frame_path)[0] + "_trial_stats.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"\n📈 통계 저장: {stats_file}")
    else:
        print("\n⚠️  통계 데이터 없음")

    # 9. 메인 파일 저장
    root = os.path.splitext(frame_path)[0]
    output_path = root + "_final.csv"

    try:
        df_frame.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 메인파일 저장: {output_path}")
    except PermissionError:
        alt_path = root + "_final_backup.csv"
        df_frame.to_csv(alt_path, index=False, encoding='utf-8-sig')
        print(f"🔄 백업 저장: {alt_path}")
    except Exception as e:
        print(f"❌ 파일 저장 에러: {e}")

    print("\n🎉 완벽 완료!")
    if stats_results:
        print("📋 통계 결과:")
        stats_df_print = pd.DataFrame(stats_results)
        print(stats_df_print[['trial', 'x_cor_aver', 'y_cor_aver', 'x_sd', 'y_sd']].to_string(index=False))


if __name__ == "__main__":
    process_frame_data(FRAME_DATA_PATH, VIDEO_PATH, TRIAL_PATH, SAVE_AS)
