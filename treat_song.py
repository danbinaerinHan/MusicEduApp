import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS 시스템에서 사용 가능한 한글 폰트들을 시도
korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Nanum Gothic', 'DejaVu Sans']

for font in korean_fonts:
    try:
        plt.rcParams['font.family'] = font
        break
    except:
        continue
        
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


NOTE_NAMES = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시']
YUL_NAMES = ['황', '대', '태', '협', '고', '중', '유', '임', '이', '남', '무', '응',]


def midi_to_note_name(midi_num): 
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12] 
    if octave <= 2:
        return f"낮은 {note}"
    elif octave == 3:
        return note
    else:
        return f"높은 {note}"

def midi_to_yul(midi_num):
    octave = (midi_num // 12) - 2
    note = YUL_NAMES[(midi_num % 12) -3]
    if octave <= 2:
        return f"배{note}"
    elif octave == 3:
        return note
    else:
        return f"청{note}"


    # --- MIDI Number to Note Name ---
class GetSongInfo:
    def __init__(self, sample, song_info):
        self.sample = sample
        self.song_id = sample['song_id']
        self.song_type = song_info[self.song_id]['song_type']
        self.answers, self.measure_indices, self.performance_data = sample['answers'], sample['measure_indices'], sample['performance_data']
        self.lyrics_per_measure = song_info[self.song_id]['lyrics_per_measure'] if self.song_type not in ['recorder', 'danso'] else None
        self.learning_points = song_info[self.song_id]['learning_points']

    # --- Pitch Difference Calculation ---
    def pitch_difference(self, answer_pitch, performed_pitches, threshold=0.5):
        if answer_pitch == -1 or not performed_pitches:
            return None
        avg_pitch = sum(performed_pitches) / len(performed_pitches)
        diff = avg_pitch - answer_pitch
        # threshold 범위 내의 차이는 0으로 처리 (정확한 것으로 간주)
        if abs(diff) < threshold:
            return 0.0
        return diff

    # --- Pitch Level Classification with Octave Consideration ---
    def classify_pitch_level(self, pitches):
        has_high = any(p >= 60 for p in pitches)
        has_mid = any(53 <= p <= 57 for p in pitches)
        has_low = any(p <= 52 for p in pitches)

        if has_high:
            return "고음역대"
        elif has_mid:
            return "중간 음역대"
        else:
            return "저음역대"

    # --- Measure-Level Musical Feature Analysis ---
    def measure_features(self, measure, lyrics_per_measure=None, threshold=1.0):
        notes = [(a, p, idx) for idx, (a, i, p) in enumerate(zip(self.answers, self.measure_indices, self.performance_data)) if i == measure and a != -1 and p]
        if not notes:
            return {}, 0.0, 0

        answer_pitches = [a for a, _, _ in notes]
        pitch_range = max(answer_pitches) - min(answer_pitches)

        if all(x == answer_pitches[0] for x in answer_pitches):
            direction = "유지"
        elif all(x <= y for x, y in zip(answer_pitches, answer_pitches[1:])):
            direction = "상행"
        elif all(x >= y for x, y in zip(answer_pitches, answer_pitches[1:])):
            direction = "하행"
        else:
            direction = "복합적 이동"

        density = len(answer_pitches)
        note_density = "빠른 패시지" if density >= 6 else ("보통" if density >= 3 else "느린 패시지")
        pitch_level = self.classify_pitch_level(answer_pitches)

        pitch_diffs = [self.pitch_difference(a, p, threshold) for a, p, _ in notes if self.pitch_difference(a, p, threshold) is not None]
        avg_diff = sum(pitch_diffs) / len(pitch_diffs) if pitch_diffs else 0
        error_count = sum(d != 0.0 for d in pitch_diffs)  # 0이 아닌 값들이 오류

        level = "정확" if avg_diff == 0.0 else ("조금 높음" if avg_diff > 0 else "조금 낮음")
        if abs(avg_diff) > 1.5:
            level = "매우 높음" if avg_diff > 0 else "매우 낮음"

        wrong_lyrics = []
        if self.song_type == 'recorder':
            # recorder 타입일 때는 note_name만 사용
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p, threshold)
                if diff != 0.0:  # threshold를 넘는 오류인 경우
                    wrong_lyrics.append(f"{midi_to_note_name(a)}")
        elif self.song_type == 'danso':
            # danso 타입일 때는 yul_name만 사용
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p, threshold)
                if diff != 0.0:  # threshold를 넘는 오류인 경우
                    wrong_lyrics.append(f"{midi_to_yul(a)}")
        elif lyrics_per_measure and str(measure) in lyrics_per_measure:
            # 다른 타입일 때는 lyrics_per_measure 사용
            syllables = lyrics_per_measure[str(measure)]
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p, threshold)
                if diff != 0.0 and local_idx < len(syllables):  # threshold를 넘는 오류인 경우
                    syllable = syllables[local_idx]
                    wrong_lyrics.append(f"{syllable}({midi_to_note_name(a)})")

        result = {
            'pitch_level': pitch_level,
            'pitch_variability': '음정 변화가 많음' if pitch_range >= 5 else '음정 변화 적음',
            'pitch_direction': direction,
            'note_density': note_density,
            'pitch_error': f"{level} ({avg_diff:+.1f})",
            'frequent_wrong_syllables': wrong_lyrics[:3]  # 최대 3개
        }
        return result, abs(avg_diff), error_count

    # --- 핵심 마디만 요약해서 LLM 입력용 데이터 생성 ---
    def __call__(self, top_n=3, threshold=0.5):
        all_measures = sorted(set(self.measure_indices))
        analysis = {}
        pitch_accurate = 0
        total = 0
        note_errors = defaultdict(list)
        error_stats = {}

        for m in all_measures:
            result, avg_err, err_count = self.measure_features(m, self.lyrics_per_measure, threshold)
            analysis[str(m)] = result
            error_stats[m] = {'avg_error': avg_err, 'error_count': err_count}

        for ans, perf in zip(self.answers, self.performance_data):
            if ans == -1 or not perf:
                continue
            total += 1
            diff = self.pitch_difference(ans, perf, threshold)
            if diff == 0.0:  # threshold 범위 내라서 0으로 처리된 경우
                pitch_accurate += 1
            else:  # threshold를 넘는 오류인 경우
                note_errors[ans].append(abs(diff))

        pitch_accuracy_percent = (pitch_accurate / total * 100) if total else 100
        top_wrong_notes = sorted(note_errors.items(), key=lambda x: (len(x[1]), sum(x[1])), reverse=True)[:3]
        frequent_wrong_notes = []
        for n, _ in top_wrong_notes:
            if self.song_type == 'danso':
                frequent_wrong_notes.append({"note": midi_to_yul(int(n))})
            else:
                frequent_wrong_notes.append({"note": midi_to_note_name(int(n))})

        top_by_error = sorted(error_stats.items(), key=lambda x: x[1]['avg_error'], reverse=True)[:top_n]
        top_by_count = sorted(error_stats.items(), key=lambda x: x[1]['error_count'], reverse=True)[:top_n]
        selected = set(str(m) for m, _ in top_by_error + top_by_count)

        critical_measures = {}
        for m in selected:
            critical_measures[m] = analysis[m]
            if self.song_type in ['recorder', 'danso']:
                # recorder, danso 타입일 때는 lyrics 추가하지 않음
                pass
            elif self.lyrics_per_measure and str(m) in self.lyrics_per_measure:
                critical_measures[m]['lyrics'] = self.lyrics_per_measure[str(m)]
            if self.learning_points and str(m) in self.learning_points:
                critical_measures[m]['learning_point'] = self.learning_points[str(m)]

        return {
            "song_info": {
                "song_id": self.song_id,
                "song_type": self.song_type
            },
            "overall_summary": {
                "pitch_accuracy_percent": round(pitch_accuracy_percent, 2),
                "frequent_wrong_notes": frequent_wrong_notes
            },
            "critical_measures": critical_measures
        }



def visualize_acc_by_type(acc_by_type):
  for song_type, accs in acc_by_type.items():
    print(f"\n{song_type}:")
    print(f"  샘플 수: {len(accs)}개")
    print(f"  평균: {np.mean(accs):.2f}%")
    print(f"  중간값: {np.median(accs):.2f}%")
    print(f"  표준편차: {np.std(accs):.2f}%")
    print(f"  범위: {min(accs):.2f}% ~ {max(accs):.2f}%")

  # 시각화
  fig, axes = plt.subplots(2, 2, figsize=(15, 10))

  # 1. 히스토그램 비교
  ax1 = axes[0, 0]
  colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
  for i, (song_type, accs) in enumerate(acc_by_type.items()):
      ax1.hist(accs, bins=10, alpha=0.6, label=f'{song_type} (n={len(accs)})', 
              color=colors[i % len(colors)], edgecolor='black')
  ax1.set_title('Song Type별 정확도 히스토그램')
  ax1.set_xlabel('정확도 (%)')
  ax1.set_ylabel('빈도')
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  # 2. 박스플롯 비교
  ax2 = axes[0, 1]
  song_types = list(acc_by_type.keys())
  accs_list = [acc_by_type[st] for st in song_types]
  box_plot = ax2.boxplot(accs_list, labels=song_types, patch_artist=True)
  for patch, color in zip(box_plot['boxes'], colors):
      patch.set_facecolor(color)
  ax2.set_title('Song Type별 정확도 박스플롯')
  ax2.set_ylabel('정확도 (%)')
  ax2.grid(True, alpha=0.3)

  # 3. 평균 비교 막대그래프
  ax3 = axes[1, 0]
  means = [np.mean(acc_by_type[st]) for st in song_types]
  bars = ax3.bar(song_types, means, color=colors[:len(song_types)], alpha=0.7, edgecolor='black')
  ax3.set_title('Song Type별 평균 정확도')
  ax3.set_ylabel('평균 정확도 (%)')
  ax3.grid(True, alpha=0.3)
  # 막대 위에 수치 표시
  for bar, mean in zip(bars, means):
      ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
              f'{mean:.1f}%', ha='center', va='bottom')

  # 4. 각 타입별 구간 분포
  ax4 = axes[1, 1]
  ranges = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
  range_labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']

  width = 0.2
  x_pos = np.arange(len(range_labels))

  for i, (song_type, accs) in enumerate(acc_by_type.items()):
      counts = []
      for range_i, (start, end) in enumerate(ranges):
          if range_i == len(ranges) - 1:  # 마지막 구간은 end 값도 포함
              count = sum(1 for acc in accs if start <= acc <= end)
          else:
              count = sum(1 for acc in accs if start <= acc < end)
          percentage = count / len(accs) * 100 if len(accs) > 0 else 0
          counts.append(percentage)
      
      ax4.bar(x_pos + i * width, counts, width, label=song_type, 
              color=colors[i % len(colors)], alpha=0.7)

  ax4.set_title('Song Type별 구간 분포 (%)')
  ax4.set_xlabel('정확도 구간')
  ax4.set_ylabel('비율 (%)')
  ax4.set_xticks(x_pos + width * (len(song_types) - 1) / 2)
  ax4.set_xticklabels(range_labels)
  ax4.legend()
  ax4.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()