import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def list_all_files(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.is_file()]


def get_id_and_script(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # print(Path(file_path).parent.__str__())
    parent_dir = Path(file_path).parent.__str__().split('/')[-3:]
    parent_dir = '_'.join(parent_dir)
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    file_id = f'{parent_dir}_{file_id}'

    if len(file_id) < 10:
        print(file_id)

    speaker_id = data["05_speakerinfo"]["1_id"]
    transcription = data["06_transcription"]["1_text"]

    return file_id, speaker_id, transcription


def main(directory):
    all_files = list_all_files(directory)

    speaker_transcriptions = defaultdict(list)

    # 결과 출력
    for file_path in tqdm(all_files, ncols=75):
        file_id, speaker_id, transcription = get_id_and_script(file_path)
        speaker_transcriptions[speaker_id].append((file_id, transcription))

    for speaker_id, _ in speaker_transcriptions.items():
        speaker_transcriptions[speaker_id].sort(key=lambda x: x[0])

    # speaker_id별 파일 저장 경로 지정
    output_dir = "speaker_transcriptions"  # 저장할 디렉토리 경로
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성

    # speaker_id별로 파일 저장
    for speaker_id, transcriptions in speaker_transcriptions.items():
        file_path = os.path.join(output_dir, f"{speaker_id}.txt")  # 파일명: speaker_id.txt
        with open(file_path, "w", encoding="utf-8") as f:
            for file_id, transcription in transcriptions:
                f.write(transcription + "\n")  # 줄바꿈하여 저장


if __name__ == '__main__':
    directory = '/home/goldenmine/Desktop/Development/dataset/train_label'  # 대상 디렉토리 경로를 입력
    main(directory)