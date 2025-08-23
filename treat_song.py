import json
from collections import defaultdict, Counter
import pandas as pd

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
    def pitch_difference(self, answer_pitch, performed_pitches):
        if answer_pitch == -1 or not performed_pitches:
            return None
        avg_pitch = sum(performed_pitches) / len(performed_pitches)
        return avg_pitch - answer_pitch

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
    def measure_features(self, measure, lyrics_per_measure=None, threshold=0.5):
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

        pitch_diffs = [self.pitch_difference(a, p) for a, p, _ in notes if self.pitch_difference(a, p) is not None]
        avg_diff = sum(pitch_diffs) / len(pitch_diffs) if pitch_diffs else 0
        error_count = sum(abs(d) >= threshold for d in pitch_diffs)

        level = "정확" if abs(avg_diff) < threshold else ("조금 높음" if avg_diff > 0 else "조금 낮음")
        if abs(avg_diff) > 1.5:
            level = "매우 높음" if avg_diff > 0 else "매우 낮음"

        wrong_lyrics = []
        if self.song_type == 'recorder':
            # recorder 타입일 때는 note_name만 사용
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p)
                if diff and abs(diff) >= threshold:
                    wrong_lyrics.append(f"{midi_to_note_name(a)}")
        elif self.song_type == 'danso':
            # danso 타입일 때는 yul_name만 사용
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p)
                if diff and abs(diff) >= threshold:
                    wrong_lyrics.append(f"{midi_to_yul(a)}")
        elif lyrics_per_measure and str(measure) in lyrics_per_measure:
            # 다른 타입일 때는 lyrics_per_measure 사용
            syllables = lyrics_per_measure[str(measure)]
            for local_idx, (a, p, _) in enumerate(notes):
                diff = self.pitch_difference(a, p)
                if diff and abs(diff) >= threshold and local_idx < len(syllables):
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
            diff = self.pitch_difference(ans, perf)
            if abs(diff) < threshold:
                pitch_accurate += 1
            elif abs(diff) >= threshold:
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


