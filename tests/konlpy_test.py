from konlpy.tag import Okt

# 문서 예시
document = "인공지능은 미래 사회를 변화시키는 핵심 기술로 주목받고 있습니다. 주목 받습니다."

# Okt 형태소 분석기 초기화
okt = Okt()

morphs = okt.morphs(document)
print("추출:", morphs)


from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel

# KoBERT 모델 로드
kobert_model = AutoModel.from_pretrained("skt/kobert-base-v1")
kobert_tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

# KeyBERT 초기화
kw_model = KeyBERT(model=kobert_model)

# 키워드 추출
document = ' '.join(morphs)
keywords = kw_model.extract_keywords(document, keyphrase_ngram_range=(1, 2), top_n=5)
print("추출된 키워드:", keywords)