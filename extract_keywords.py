from pathlib import Path

from PyKomoran import Komoran
from keybert import KeyBERT
from konlpy.tag import Okt
from tqdm import tqdm
from transformers import BertModel
from collections import defaultdict

komoran = Komoran("EXP")  # OR EXP


def filter_text(text):
    def filter_sw(string):
        split = string.split("/")

        if len(split) == 1:
            return text

        # return True
        if len(split) > 2:
            word = '/'.join(split[:-1])
            wtype = split[-1]
        else:
            word, wtype = split

        if wtype == 'NNP' or wtype == 'NNG':
            return True

        if wtype == 'VV' or wtype == 'VA':
            return True

        return False

    # text = self.clean_text(text)
    text = text.replace("/", " ")

    text = (text.replace("#@이름#", "#")
            .replace(" # ", "#")
            .replace("##", "#")
            .replace('즺', '짖')
            .replace('즵', '집')
            .replace('즫', '짇')
            .replace('즥', '직')
            .replace('즷', '짓')
            .replace('즴', '짐')
            .replace('즨', '진')
            .replace('즹', '징')
            .replace('즬', '질')
            .replace('즿', '짛')
            .replace('즼', '짘')
            .replace('즽', '짙')
            .replace('즻', '짗')
            .replace('즾', '짚')
            .replace('즤', '지')
            )

    res = komoran.get_plain_text(text).split(' ')

    # res = list(filter(filter_sw, res))  # 필터링
    res = list(map(lambda x: x.split('/')[0], res))  # 대한민국/NNP 같은 단어가 있으면 슬래시 뒤 문자 떼버림

    return res


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
        "ㅁ", "기",

        "으로써", "로써", "로서", "으로서",
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
    # okt = Okt()

    # 모든 문서에 대한 형태소 분석
    print("morphing...")
    all_morphs = []
    for path in tqdm(list_all_files('speaker_transcriptions'), ncols=75):
        with open(path, 'rt') as file:
            text = file.read()
        # text = okt.morphs(text)
        text = filter_text(text)
        text = ' '.join(text)
        all_morphs.append(text)

    # all_morphs = ['\n'.join(all_morphs)]

    stop_words = get_stop_words()

    # 키워드 추출
    def extract_keywords(ngram_range=(1, 1), output_file='result.txt'):
        print(f"extracting keywords... {ngram_range} {output_file}")
        all_keywords = []
        for morphs in tqdm(all_morphs, ncols=75):
            keywords = kw_model.extract_keywords(
                morphs,
                keyphrase_ngram_range=ngram_range,
                stop_words=stop_words,
                top_n=50
            )
            all_keywords.append(keywords)

        # 키워드 병합
        merged_keywords = defaultdict(float)
        for doc_keywords in all_keywords:
            for keyword, score in doc_keywords:
                merged_keywords[keyword] += score  # 중요도 합산

        # 중요도 기준으로 정렬
        sorted_keywords = sorted(merged_keywords.items(), key=lambda x: x[1], reverse=True)

        # 상위 50%
        top50 = len(sorted_keywords) // 100 * 50
        top70 = len(sorted_keywords) // 100 * 70

        # 키워드 문자열로 변환
        keywords = list(map(lambda x: f'{x[0]}, {x[1]}', sorted_keywords[:top50]))
        keywords2 = list(map(lambda x: f'{x[0]}, {x[1]}', sorted_keywords[top50:top70]))

        # keywords = [text.replace(' ', '') for text in keywords]

        # result.txt 파일에 개행 단위로 저장
        with open(f'{output_file}_top0.txt', 'wt', encoding='utf-8') as file:
            file.write('\n'.join(keywords))  # 문자열 리스트를 개행 단위로 연결 후 파일에 쓰기

        with open(f'{output_file}_top50.txt', 'wt', encoding='utf-8') as file:
            file.write('\n'.join(keywords2))  # 문자열 리스트를 개행 단위로 연결 후 파일에 쓰기

    extract_keywords((1, 1), 'result_11')
    # extract_keywords((1, 2), 'result_12_okt.txt')
    # extract_keywords((2, 2), 'result_22_okt.txt')


if __name__ == '__main__':
    main()