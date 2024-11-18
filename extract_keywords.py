from pathlib import Path

from PyKomoran import Komoran
from keybert import KeyBERT
from kobert_tokenizer import KoBERTTokenizer
from konlpy.tag import Okt
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
from collections import defaultdict


def list_all_files(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.is_file()]


def get_stop_words():

    josa_list = [
        "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
        "에게", "한테", "에게서", "한테서", "까지", "부터", "와", "과",
        "보다", "아", "야", "은", "는", "도", "만", "까지만", "마저",
        "조차", "뿐", "처럼", "같이", "만큼", "이에", "밖에"
    ]

    korean_eomi_list = [
        # 종결형 어미
        "다", "요", "네", "군", "나", "까", "냐", "니", "라", "어라", "아라", "자", "읍시다", "ㅂ시다",

        # 대등 연결 어미
        "고", "나", "지만", "면",

        # 종속 연결 어미
        "서", "니", "라서", "느라고", "느라", "면서", "으므로", "으니까", "으며", "으려면", "으려고",

        # 관형사형 어미
        "ㄴ", "는", "ㄹ", "던",

        # 명사형 어미
        "ㅁ", "기"
    ]

    stop_words = []
    stop_words.extend(josa_list)
    stop_words.extend(korean_eomi_list)

    return stop_words

def main():
    model_name = "skt/kobert-base-v1"
    # tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)

    # KeyBERT 초기화
    kw_model = KeyBERT(model=model)

    # Okt 형태소 분석기 초기화
    okt = Okt()

    # 모든 문서에 대한 형태소 분석
    print("morphing...")
    all_morphs = []
    for path in tqdm(list_all_files('speaker_transcriptions')[:2], ncols=75):
        with open(path, 'rt') as file:
            text = file.read()
        # text = okt.morphs(text)
        # text = ' '.join(morphs)
        all_morphs.append(text)

    stop_words = get_stop_words()

    # 키워드 추출
    print("extracting keywords...")
    all_keywords = []
    for morphs in tqdm(all_morphs, ncols=75):
        keywords = kw_model.extract_keywords(morphs, keyphrase_ngram_range=(1, 2), stop_words=stop_words, top_n=100)
        all_keywords.append(keywords)

    # 키워드 병합
    merged_keywords = defaultdict(float)
    for doc_keywords in all_keywords:
        for keyword, score in doc_keywords:
            merged_keywords[keyword] += score  # 중요도 합산

    # 중요도 기준으로 정렬
    sorted_keywords = sorted(merged_keywords.items(), key=lambda x: x[1], reverse=True)
    keywords = list(map(lambda x: f'{x[0]}, {x[1]}', sorted_keywords))
    # keywords = [text.replace(' ', '') for text in keywords]

    # result.txt 파일에 개행 단위로 저장
    with open("result.txt", "wt", encoding="utf-8") as file:
        file.write('\n'.join(keywords))  # 문자열 리스트를 개행 단위로 연결 후 파일에 쓰기


if __name__ == '__main__':
    main()