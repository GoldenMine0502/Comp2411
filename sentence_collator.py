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

    speaker_id = data["05_speakerinfo"]["1_id"]
    transcription = data["06_transcription"]["1_text"]

    return speaker_id, transcription


def main():
    directory = '/Users/taewonkim/Downloads/137.한국어 대학 강의 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL'  # 대상 디렉토리 경로를 입력
    all_files = list_all_files(directory)

    speaker_transcriptions = defaultdict(list)

    # 결과 출력
    for file_path in tqdm(all_files, ncols=75):
        speaker_id, transcription = get_id_and_script(file_path)
        speaker_transcriptions[speaker_id].append(transcription)


if __name__ == '__main__':
    main()