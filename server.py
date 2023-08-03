import numpy as np
import keras
import json
import tensorflow_addons as tfa
from flask import Flask, request, jsonify
from keras_preprocessing.text import tokenizer_from_json
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

app = Flask(__name__)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = keras.models.load_model("BERT_model.h5", custom_objects={"TFBertModel": TFBertModel})

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)

    input_ids = []
    attention_masks = []
    for sent in tqdm(data['text']):
        bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                              return_attention_mask=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)

    prediction = model.predict([input_ids, attention_masks])

    answers = []
    for x in prediction:
        if x[0] > x[1] and x[0] > x[2]:
            answers.append("negative")
        elif x[1] > x[0] and x[1] > x[2]:
            answers.append("neutral")
        else:
            answers.append("positive")
    return jsonify(answers)

if __name__ == '__main__':
    app.run(port=5000, debug=True)