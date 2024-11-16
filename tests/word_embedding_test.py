from transformers import BertModel, BertTokenizer
import torch


def get_kobert_embeddings(texts, model_name='skt/kobert-base-v1'):
    """
    KoBERT를 사용하여 텍스트 임베딩을 추출합니다.

    Args:
        texts (list of str): 입력 텍스트 리스트.
        model_name (str): KoBERT 모델 이름 (기본값: skt/kobert-base-v1).

    Returns:
        torch.Tensor: 문장 임베딩 (batch_size x embedding_dim).
    """
    # KoBERT 토크나이저 및 모델 로드
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 입력 토큰화
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # 모델을 통해 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)

    # 문장 임베딩은 [CLS] 토큰의 출력 벡터로 추출
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings


# 테스트 데이터
texts = ["자연어 처리는 정말 재미있는 분야입니다.", "KoBERT를 활용한 임베딩 추출 방법을 배워봅시다."]
embeddings = get_kobert_embeddings(texts)

# 결과 출력
print("Embedding Shape:", embeddings.shape)  # (batch_size, embedding_dim)
print("Embedding Example:", embeddings[0])