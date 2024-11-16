import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
tokenizer = PreTrainedTokenizerFast.from_pretrained('Soyoung97/gec_kr')
model = BartForConditionalGeneration.from_pretrained('Soyoung97/gec_kr')
text = '토끼와 거북이는 틀리다.'
raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
corrected_ids = model.generate(torch.tensor([input_ids]),
                                max_length=128,
                                eos_token_id=1, num_beams=4,
                                early_stopping=True, repetition_penalty=2.0)
output_text = tokenizer.decode(corrected_ids.squeeze().tolist(), skip_special_tokens=True)
print(output_text)