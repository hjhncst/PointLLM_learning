import argparse
import json
import os
import random
random.seed(0)

import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch


import numpy as np
from tqdm import tqdm

class TraditionalMetricEvaluator():
    def __init__(self, inputs, output_dir, output_file):

        # 初始化评估器，接收输入数据、输出目录和输出文件名
        self.results = inputs['results']  # 评估结果数据
        self.inference_prompt = inputs['prompt']  # 推理提示
        self.output_dir = output_dir  # 输出目录
        self.output_file = output_file  # 输出文件名
        self.rouge = Rouge()  # 初始化ROUGE评估工具
        self.response_data = []  # 存储每个结果的详细信息

        self.ground_truths = []  # 存储真实标签
        self.generated_captions = []  # 存储生成的标题

        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')  # 初始化SBERT模型

        # 初始化SimCSE模型和tokenizer
        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

        # 初始化评分字典，存储不同指标的评分列表
        self.scores = {
            'bleu-1': [],
            'bleu-2': [],
            'bleu-3': [],
            'bleu-4': [],
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'meteor': [],
            'sbert_similarity': [],
            'simcse_similarity': []
        }

    def evaluate_result(self, result):

        # 评估单个结果
        object_id = result['object_id']  # 获取对象ID
        ground_truth = result['ground_truth']  # 获取真实标签
        model_output = result['model_output']  # 获取模型输出

        if model_output == "":
            # * all score should be 0
            model_output = "##"

        # create a SmoothingFunction object
        smoothing_function = SmoothingFunction().method1 # * used to deal with non-overlap n-gram

        # calculate BLEU-1 score with smoothing function
        bleu_1_score = sentence_bleu([ground_truth.split()], model_output.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)

        # calculate BLEU-2, BLEU-3, and BLEU-4 scores
        bleu_2_score = sentence_bleu([ground_truth.split()], model_output.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        bleu_3_score = sentence_bleu([ground_truth.split()], model_output.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
        bleu_4_score = sentence_bleu([ground_truth.split()], model_output.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

        # calculate ROUGE-L score
        rouge_scores_l = self.rouge.get_scores(model_output, ground_truth)[0]['rouge-l']
        rouge_scores_1 = self.rouge.get_scores(model_output, ground_truth)[0]['rouge-1']
        rouge_scores_2 = self.rouge.get_scores(model_output, ground_truth)[0]['rouge-2']

        # calculate METEOR score
        meteor_scores = meteor_score([ground_truth.split()], model_output.split())

        # Calculate SBERT similarity
        embeddings = self.sbert_model.encode([ground_truth, model_output])
        sbert_similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()

        # calculate SimCSE similarity
        # Tokenize input texts
        inputs = self.simcse_tokenizer([ground_truth, model_output], padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarity
        simcse_similarity = 1 - cosine(embeddings[0], embeddings[1]) # * consine actually calculates consine distance, which is 1 - consine similarity

        scores = {
            'bleu-1': bleu_1_score * 100,
            'bleu-2': bleu_2_score * 100,
            'bleu-3': bleu_3_score * 100,
            'bleu-4': bleu_4_score * 100,
            'rouge-l': rouge_scores_l['f'] * 100,
            'rouge-1': rouge_scores_1['f'] * 100,
            'rouge-2': rouge_scores_2['f'] * 100,
            'meteor': meteor_scores * 100,
            'sbert_similarity': sbert_similarity * 100,
            'simcse_similarity': simcse_similarity * 100
        }

        return object_id, model_output, ground_truth, scores

    def evaluate(self):
        # 打印开始评估的提示信息
        print("Starting evaluation...")

        # 使用tqdm库显示评估进度条，遍历self.results中的每个结果
        for result in tqdm(self.results, desc="Evaluating"):  
            # 调用evaluate_result方法，获取评估结果中的object_id, model_output, ground_truth, scores
            object_id, model_output, ground_truth, scores = self.evaluate_result(result)

            # save the object_id, model_output, ground_truth, and scores for each result
            self.response_data.append({
                'object_id': object_id,
                'ground_truth': ground_truth,
                'model_output': model_output,
                'scores': scores,
            })

            # save the scores for overall results
            for metric, score in scores.items():
                self.scores[metric].append(score)
        
        print("Evaluation finished.")
        self.save_results()
        self.print_results()

    def save_results(self):
        # 构建输出文件的完整路径，路径由输出目录和输出文件名组成
        output_path = os.path.join(self.output_dir, self.output_file)

        # 以写模式打开输出文件
        with open(output_path, 'w') as f:
            # 准备要保存的结果数据
            results_to_save = {
                # 保存推理提示
                'inference_prompt': self.inference_prompt,
                # 保存各项指标的平均分数，分数保留四位小数
                'overall_scores': {metric: f"{np.mean(scores):.4f}" for metric, scores in self.scores.items()},
                # 保存响应数据
                'results': self.response_data,
            }
            # 将结果数据以JSON格式写入文件，缩进为2个空格
            json.dump(results_to_save, f, indent=2)
        
        # 打印保存结果的文件路径
        print(f"Results saved to {output_path}")

    def print_results(self):  # 定义一个名为print_results的方法，用于打印结果
        print('-' * 80)  # 打印80个连字符，用于分隔输出内容
        print("Results:")  # 打印"Results:"，表示接下来将显示结果
        for metric, scores in self.scores.items():  # 遍历self.scores字典，其中metric是键，scores是值
            print(f"Average {metric.upper()} Score: {np.mean(scores):.4f}")  # 打印每个指标的均值分数

def start_evaluation(results, output_dir, output_file,
                        parallel=True, num_workers=20):
    """
    开始评估函数，用于对结果进行评估并保存评估结果。
    Args:
        results: dict or file path to the json file containing the dict
        output_file: the path the final evaluation results to be saved.
    """
    if isinstance(results, str):
        with open(results, 'r') as fp:
            results = json.load(fp)

    evaluator = TraditionalMetricEvaluator(results, output_dir, output_file) 
    evaluator.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, \
                        default="", help="Path to the results file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_path)

    output_file = os.path.basename(args.results_path).replace(".json", f"_evaluated_traditional.json")

    start_evaluation(results=args.results_path, output_dir=args.output_dir, output_file=output_file)
    