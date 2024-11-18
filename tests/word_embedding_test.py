from PyKomoran import Komoran
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, BertTokenizer
import torch


komoran = Komoran("EXP")  # OR EXP


def filter_text(text):
    def filter_sw(string):
        split = string.split("/")

        # if len(split) == 1:
        #     return '#' not in text

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

    res = komoran.get_plain_text(text).split(' ')

    res = list(filter(filter_sw, res))  # 필터링
    res = list(map(lambda x: x.split('/')[0], res))  # 대한민국/NNP 같은 단어가 있으면 슬래시 뒤 문자 떼버림

    return res

def get_kobert_embeddings(texts, model_name='skt/kobert-base-v1', top_n=5):
    # KoBERT 토크나이저 및 모델 로드
    tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)

    text = " ".join(filter_text(" ".join(texts)))
    print('text:', text)

    # 입력 토큰화
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # 모델을 통해 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)

    # 마지막 층의 Attention 가중치 가져오기
    attentions = outputs.attentions[-1]  # 마지막 레이어의 Attention
    mean_attention = attentions.mean(dim=1)  # 모든 헤드 평균

    # 입력 텍스트의 토큰 ID 및 매핑
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # CLS 토큰 제외
    tokens = tokens[1:-1]
    print('tokens:', tokens)
    mean_attention = mean_attention[0][0, 1:-1]  # 첫 문장에 대한 [CLS]와 [SEP] 제거

    # 중요도가 높은 토큰 정렬
    top_indices = torch.argsort(mean_attention, descending=True)[:top_n]
    keywords = [tokens[idx] for idx in top_indices]

    return keywords


# 테스트 데이터
texts = ["자연어 처리는 정말 재미있는 분야입니다.", "KoBERT를 활용한 임베딩 추출 방법을 배워봅시다.", "자연어 처리는 재밌습니다.", "처리 재밌다", "처리"]
keywords = get_kobert_embeddings(texts)

print(keywords)