from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

def calculate_accuracy(correct_list):
    # 计算准确率
    accuracy = sum(correct_list) / len(correct_list)
    return accuracy


def format_label(label):
    if isinstance(label, str) and label.startswith('LABEL_'):
        return int(label.split('_')[-1])
    return label


def v_entropy(data_fn, model, tokenizer, input_key='sentence1', batch_size=100):
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True,
                          device=device)
    data = pd.read_csv(data_fn)

    entropies = []
    correct = []
    predicted_labels = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j + batch_size]
        predictions = classifier(batch[input_key].tolist())

        for i in range(len(batch)):
            matches = [d for d in predictions[i] if format_label(d['label']) == format_label(batch.iloc[i]['label'])]
            prob = matches[0]['score'] if matches else 1e-10
            entropies.append(-1 * np.log2(prob))

            predicted_label = format_label(max(predictions[i], key=lambda x: x['score'])['label'])
            predicted_labels.append(predicted_label)
            correct.append(predicted_label == format_label(batch.iloc[i]['label']))

    torch.cuda.empty_cache()
    return entropies, correct, predicted_labels


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='sentence1'):
    data = pd.read_csv(data_fn)
    data['H_yb'], _, _ = v_entropy(null_data_fn, null_model, tokenizer, input_key=input_key)
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(data_fn, model, tokenizer,
                                                                          input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']


    # 计算准确率
    accuracy = calculate_accuracy(data['correct_yx'])
    print(f"Accuracy: {accuracy:.4f}")  # 打印准确率

    if out_fn:
        data.to_csv(out_fn)

    return data


if __name__ == "__main__":
    os.makedirs('PVI', exist_ok=True)

    parser.add_argument('--data_dir', help='Data directory', required=True, type=str)
    parser.add_argument('--model_dir', help='Model directory', required=True, type=str)
    parser.add_argument('--output_dir', help='Output directory for results', default='PVI', type=str)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_DIR = args.model_dir
    OUTPUT_DIR = args.output_dir

    tokenizer_name = 'hfl/chinese-bert-wwm'
    model_name = 'hfl/chinese-bert-wwm'

    # OCNLI数据集
    print(model_name, 'ocnli')
    v_info(f"{DATA_DIR}/ocnli_train.csv", f"{MODEL_DIR}/{model_name}_ocnli",
           f"{DATA_DIR}/ocnli_null.csv", f"{MODEL_DIR}/{model_name}_ocnli_null",
           tokenizer_name, out_fn=f"{OUTPUT_DIR}/{model_name}_ocnli_train.csv")

    v_info(f"{DATA_DIR}/ocnli_validation.csv", f"{MODEL_DIR}/{model_name}_ocnli",
           f"{DATA_DIR}/ocnli_null.csv", f"{MODEL_DIR}/{model_name}_ocnli_null",
           tokenizer_name, out_fn=f"{OUTPUT_DIR}/{model_name}_ocnli_validation.csv")
