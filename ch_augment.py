import os
import pandas as pd
import random
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()

class OCNLITransformation(object):
    """
    Parent class for transforming the OCNLI input to extract specific input attributes 
    (e.g., just the hypothesis by leaving out the premise). Also reformats the input 
    into a single string. The transformed data is saved as a CSV.
    """
    def __init__(self, name, output_dir, train_size=1.0):
        """
        Args:
            name: Transformation name
            output_dir: where to save the CSV with the transformed attribute
            train_size: fraction of the training data to use
        """
        
        data_files = {
          'train': '/content/drive/MyDrive/ch_dataset_difficulty/data/ocnli_train_std.csv',
          'test': '/content/drive/MyDrive/ch_dataset_difficulty/data/ocnli_test_std.csv'
        }
        dataset = load_dataset('csv', data_files=data_files)

        # 过滤掉 label 为 -1 的数据
        self.train_data = dataset['train'].filter(lambda x: x['label'] != -1)
        self.test_data = dataset['test'].filter(lambda x: x['label'] != -1)
        
        
        
        
        #self.train_data = load_dataset('csv', data_files='./data/ocnli_train_std.csv')['train']
        #self.test_data = load_dataset('csv', data_files='./data/ocnli_test_std.csv')['test']
        
        #self.train_data = load_dataset('ocnli', split='train').filter(lambda x: x['label'] != -1)
        #self.test_data = load_dataset('ocnli', split='test').filter(lambda x: x['label'] != -1)
        self.name = name
        self.output_dir = output_dir
        self.train_size = train_size
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm')

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to OCNLI')

        if self.train_size < 1:
            train_data = self.train_data.train_test_split(train_size=self.train_size)['train']
        else:
            train_data = self.train_data

        train_data.map(self.transformation).to_pandas().to_csv(
            os.path.join(self.output_dir, f'ocnli_train_{self.name}.csv'), index=False)
        self.test_data.map(self.transformation).to_pandas().to_csv(
            os.path.join(self.output_dir, f'ocnli_test_{self.name}.csv'), index=False)


class OCNLIStandardTransformation(OCNLITransformation):
    def __init__(self, output_dir, train_size=1):
        super().__init__('standard', output_dir, train_size=train_size)

    def transformation(self, example):
        example['sentence1'] = f"前提: {example['premise']} 假设: {example['hypothesis']}"
        return example


class OCNLIHypothesisOnlyTransformation(OCNLITransformation):
    def __init__(self, output_dir):
        super().__init__('hypothesis_only', output_dir)

    def transformation(self, example):
        example['sentence1'] = f"假设: {example['hypothesis']}"
        return example


class OCNLIPremiseOnlyTransformation(OCNLITransformation):
    def __init__(self, output_dir):
        super().__init__('premise_only', output_dir)

    def transformation(self, example):
        example['sentence1'] = f"前提: {example['premise']}"
        return example


class OCNLIRawOverlapTransformation(OCNLITransformation):
    def __init__(self, output_dir):
        super().__init__('raw_overlap', output_dir)

    def transformation(self, example):
        hypothesis_tokens = self.tokenizer.tokenize(example['hypothesis'])
        premise_tokens = self.tokenizer.tokenize(example['premise'])
        overlap = set(hypothesis_tokens) & set(premise_tokens)
        hypothesis = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in hypothesis_tokens])
        premise = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in premise_tokens])
        example['sentence1'] = f"前提: {premise} 假设: {hypothesis}"
        return example


class OCNLIShuffleTransformation(OCNLITransformation):
    def __init__(self, output_dir):
        super().__init__('shuffled', output_dir)

    def transformation(self, example):
        """
        Randomly reorder the words in the hypothesis and premise.
        """
        hyp = self.tokenizer.tokenize(example['hypothesis'])
        random.shuffle(hyp)
        hyp = self.tokenizer.convert_tokens_to_string(hyp)

        prem = self.tokenizer.tokenize(example['premise'])
        random.shuffle(prem)
        prem = self.tokenizer.convert_tokens_to_string(prem)

        example['sentence1'] = f"前提: {prem} 假设: {hyp}"
        return example


class OCNLILengthTransformation(OCNLITransformation):
    def __init__(self, output_dir):
        super().__init__('length', output_dir)

    def transformation(self, example):
        hyp = ' '.join(['#'] * len(self.tokenizer.tokenize(example['hypothesis'])))
        prem = ' '.join(['#'] * len(self.tokenizer.tokenize(example['premise'])))
        example['sentence1'] = f"前提: {prem} 假设: {hyp}"
        return example


class OCNLINullTransformation(OCNLITransformation):
    def __init__(self, output_dir, train_size=1, suffix=''):
        super().__init__(f'null{suffix}', output_dir, train_size=train_size)

    def transformation(self, example):
        # 设置 sentence1 为空字符串，用于空输入转换
        example['sentence1'] = " "
        return example



if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    parser.add_argument('--output_dir', help='Output directory', required=True, type=str)
    parser.add_argument('--train_size', help='Fraction of training data to use', default=1.0, type=float)
    args = parser.parse_args()

    output_dir = args.output_dir
    train_size = args.train_size

    # Run transformations
    OCNLIStandardTransformation(output_dir, train_size).transform()
    OCNLIHypothesisOnlyTransformation(output_dir).transform()
    OCNLIPremiseOnlyTransformation(output_dir).transform()
    OCNLIRawOverlapTransformation(output_dir).transform()
    OCNLIShuffleTransformation(output_dir).transform()
    OCNLILengthTransformation(output_dir).transform()
    OCNLINullTransformation(output_dir, train_size).transform()
