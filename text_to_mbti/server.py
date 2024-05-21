import re
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# BERT tokenizer 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_text(word):
    word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', word)
    word = re.sub('\|\|\|',' ',word)
    return word    

# 텍스트 인코딩 함수
def encode_texts(texts, tokenizer, max_length=128):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    return encoded['input_ids'], encoded['attention_mask']

@app.route('/predict_mbti', methods=['POST'])
def predict_mbti():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    original_text = file.read().decode('utf-8')
    processed_text = process_text(original_text)

    # Encode the texts
    X_input_ids, X_attention_masks = encode_texts(processed_text, tokenizer)
    
    # GPU로 데이터 이동
    X_input_ids = X_input_ids.to(device)
    X_attention_masks = X_attention_masks.to(device)
    
    personality_types = ['IE', 'NS', 'FT', 'JP']
    probabilities = {
        "energy": 0,
        "recognition": 0,
        "decision": 0,
        "lifeStyle": 0
    }

    for personality_type in personality_types:
        print(f"\nTesting model for personality type {personality_type}...")
        
        # 모델 로드
        bert_model = BertForSequenceClassification.from_pretrained(f'./bert_model_{personality_type}')
        bert_model.to(device)
        bert_model.eval()

        # 예측 수행
        with torch.no_grad():
            outputs = bert_model(X_input_ids, attention_mask=X_attention_masks)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_probs = probs.cpu().numpy().flatten()
            
        print(f"Probabilities for {personality_type}: {predicted_probs}")

        if personality_type == 'IE':
            probabilities["energy"] = predicted_probs[1] * 100  # E의 확률
        elif personality_type == 'NS':
            probabilities["recognition"] = predicted_probs[1] * 100  # S의 확률
        elif personality_type == 'FT':
            probabilities["decision"] = predicted_probs[1] * 100  # T의 확률
        elif personality_type == 'JP':
            probabilities["lifeStyle"] = predicted_probs[1] * 100  # P의 확률

    return jsonify(probabilities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
