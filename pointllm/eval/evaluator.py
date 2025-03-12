import argparse
import json
import os
from utils import OpenAIGPT
from tqdm import tqdm
from multiprocessing import Pool
import random
random.seed(0)
import re

gpt4_open_free_from_cls_prompt = """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the type of object, not attributes such as color, size, or shape. Respond with 'T' if they refer to the same thing and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Example:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
Output: T#Both refer to a staircase.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """ # * about 230 input tokens

chatgpt_close_set_cls_prompt = """Given the following free-form description of a 3D object, please determine the most probable class index from the following 40 available categories, even if the description doesn't clearly refer to any one of them. Make your best-educated guess based on the information provided. If the description already contains a valid index, then the index should be selected. If it contains more than one valid index, then randomly select one index (specify your reason). If there is no valid index and it cannot be inferred from the information, return '-1#NA#Cannot infer'.
Categories:
{candidate_lists}
Reply with the format of 'index#class#short reason (no more than 10 words)'.

Examples:
Input: This is a 3D object model of a cartoon white truck.
Output: 7#car#Closest match to 'car' in categories.

Input: A green leaf in a flower pot.
Output: 26#plant#The primary subject 'leaf' directly indicates a plant.

Input: It's difficult to determine the exact type of this object due to insufficient details. But it seems to be like a piece of furniture.
Output: 33#table#Randomly select one kind of furniture from the list.

Input:  I cannot determine the specific type of the object without additional information or context.
Output: -1#NA#Cannot infer.

Now analyze the following:
Input: """

gpt4_object_captioning_prompt = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score.

Provide your score (0-100) and a short justification (less than 15 words) in the format of 'score#reason'

Example:
Human: A white brown skeleton
Model: This is a 3D model of a small, cartoon-like robot. It has a spherical body and is covered in a layer of white dust.
Output: 50#mention white; skeleton and robot have similar appearence.

Now score the following:
Human: {ground_truth}
Model: {model_output}
Output: """

chatgpt_object_captioning_prompt = gpt4_object_captioning_prompt
chatgpt_open_free_from_cls_prompt = gpt4_open_free_from_cls_prompt
gpt4_close_set_cls_prompt = chatgpt_close_set_cls_prompt

GPT_PRICES = {
    # * check https://openai.com/pricing for updated price
    "gpt-3.5-turbo-0613": {
        "price_1k_prompt_tokens": 0.0015,
        "price_1k_completion_tokens": 0.002
    },
    "gpt-3.5-turbo-1106": {
        "price_1k_prompt_tokens": 0.0010,
        "price_1k_completion_tokens": 0.002
    },
    "gpt-4-0613":{
        "price_1k_prompt_tokens": 0.03,
        "price_1k_completion_tokens": 0.06  
    },
    "gpt-4-1106-preview":{
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03
    }
}

class OpenAIOpenFreeFormClsEvaluator():
    def __init__(self, inputs, output_dir, output_file, model_type="gpt-4-0613"):
        """
        初始化OpenAIOpenFreeFormClsEvaluator类。
        Args:
            inputs: A dictionary containing the results of the evaluation. It contains two keys: "results" and "prompt".
                "prompt": str
                "results": [
                    {
                        "object_id": str,
                        "model_output": str,
                        "ground_truth": str
                    }
                ]
        """
        print("-" * 80)
        print("Initializing OpenAIEvaluator...")
        self.results = inputs['results']# * contains two keys: "results" and "prompt"
        self.inference_prompt = inputs['prompt'] # * used to prompt PointLLM
        self.correct_predictions = 0  
        self.total_predictions = 0 
        self.invalid_responses = 0
        self.response_data = [] # to save all the response data by openaigpt
        self.model_type = model_type
        self.check_model_type()

        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.default_chat_parameters = {
            "model": model_type,
            "temperature": 1, 
            "top_p": 1, 
            "max_tokens": 2048
        }

        # * price
        self.price_1k_prompt_tokens = GPT_PRICES[model_type]["price_1k_prompt_tokens"]
        self.price_1k_completion_tokens = GPT_PRICES[model_type]["price_1k_completion_tokens"]

        print(f"OpenAIGPT config: ")
        print(self.default_chat_parameters)
        
        self.openaigpt = OpenAIGPT(**self.default_chat_parameters)
        self.gpt_prompt = chatgpt_open_free_from_cls_prompt if "gpt-3.5" in model_type else gpt4_open_free_from_cls_prompt
        self.output_dir = output_dir
        self.output_file = output_file
        self.temp_output_file = self.output_file.replace(".json", "_processed_temp.json")
    
    def check_model_type(self):
        # * warning if not using gpt-4, recommend using gpt-4 for this task
        if "gpt-4" not in self.model_type:
            print(f"[WARNING] You are using {self.model_type} for evaluation. We recommend using gpt-4 for this task.")

    def resume_processing(self):
        # 构建已处理结果的文件路径
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        # 检查文件是否存在
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.correct_predictions = saved_results["correct_predictions"]
            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")
        
    def remove_temp_file(self):
        # 构建临时文件的完整路径，路径由output_dir和temp_output_file拼接而成
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        # 检查临时文件是否存在
        if os.path.exists(processed_results_path):
            # 如果存在，则删除该文件
            os.remove(processed_results_path)
            # 打印分隔线，用于美化输出
            print("-" * 80)
            # 打印提示信息，告知用户临时文件已被删除，并显示文件路径
            print(f"Removed Temporary file {processed_results_path}")

    def parse_gpt_response_evaluate(self, gpt_response):
        # 去除GPT响应字符串两端的空白字符
        gpt_response = gpt_response.strip()

        # 获取响应的第一个字符并转换为大写，作为分类结果
        cls_result = gpt_response[0].upper()
        # 获取响应中第二个字符及其后的所有字符作为原因说明，如果响应长度小于等于2，则原因为空字符串
        reason = gpt_response[2:] if len(gpt_response) > 2 else ""

        # 检查分类结果是否为'T'或'F'，如果不是，则记录无效响应并返回0, "INVALID", 和原始响应
        if cls_result not in ['T', 'F']:
            self.invalid_responses += 1
            return 0, "INVALID", gpt_response

        # 如果分类结果为'T'，则准确度为1，否则为0
        accuracy = 1 if cls_result == 'T' else 0

        # 返回准确度、分类结果和原因说明
        return accuracy, cls_result, reason

    def evaluate_result(self, result):
        # 从结果字典中提取对象ID
        object_id = result['object_id']
        # 提取真实标签
        ground_truth = result['ground_truth']
        # 提取模型输出
        model_output = result['model_output']
        # 构造消息列表，包含用户角色和格式化的提示信息
        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]

        # 使用OpenAI GPT模型安全地完成聊天，返回GPT的响应
        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 

        # 从GPT响应中提取提示令牌数
        prompt_tokens = gpt_response['usage']['prompt_tokens']
        # 从GPT响应中提取完成令牌数
        completion_tokens = gpt_response['usage']['completion_tokens']

        # 从GPT响应中提取实际的消息内容
        gpt_response = gpt_response['choices'][0]["message"]['content']


        accuracy, cls_result, reason = self.parse_gpt_response_evaluate(gpt_response) # return 0, "INVALID", gpt_response if not valid

        return object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens

    def evaluate(self):

        # 调用resume_processing方法，可能是为了恢复之前暂停的处理
        self.resume_processing()
        
        # 打印分隔线
        print('-' * 80)
        # 打印开始单线程评估的信息
        print("Starting single-thread evaluation...")
        # 获取结果列表
        results = self.results

        try:
            # 使用tqdm库对结果进行迭代，显示进度条
            for result in tqdm(results):  
                # 调用evaluate_result方法对每个结果进行评估，返回多个值
                object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens = self.evaluate_result(result)
                # 累加正确的预测数量
                self.correct_predictions += accuracy
                # 累加总的预测数量
                self.total_predictions += 1
                # 累加提示词的数量
                self.prompt_tokens += prompt_tokens
                # 累加完成词的数量
                self.completion_tokens += completion_tokens

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'model_output': model_output,
                    'gpt_cls_result': cls_result,
                    'gpt_reason': reason
                })
            
            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def parallel_evaluate(self, num_workers=20):

        # 定义一个方法用于并行评估，默认使用20个工作进程
        self.resume_processing()
        
        # 调用resume_processing方法，可能是用于恢复之前的处理状态
        print('-' * 80)
        # 打印80个连字符，用于分隔输出
        print("Starting parallel evaluation...")
        # 打印提示信息，表示开始并行评估
        results = self.results

        # 获取评估结果列表
        try:
            # 使用try-except结构来捕获可能的异常
            with Pool(num_workers) as pool:
                # 创建一个进程池，指定工作进程数量
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    # 使用tqdm创建一个进度条，总长度为结果列表的长度
                    for object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        # 使用imap_unordered方法并行处理结果列表，每次处理一个结果
                        # evaluate_result方法返回多个值，分别赋给对应的变量
                        self.correct_predictions += accuracy
                        # 累加正确的预测数量
                        self.total_predictions += 1
                        # 累加总预测数量
                        self.prompt_tokens += prompt_tokens
                        # 累加提示令牌数量
                        self.completion_tokens += completion_tokens

                        # 累加完成令牌数量
                        if cls_result == 'INVALID':
                            # 如果分类结果是'INVALID'
                            self.invalid_responses += 1

                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            'gpt_cls_result': cls_result,
                            'gpt_reason': reason
                        })

                        pbar.update()  # update the progress bar

            print("Parallel evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()

        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def save_results(self, is_temp=False):
        # 根据是否为临时保存，确定输出文件路径
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        # 计算准确率，如果总预测数减去无效响应数为0，则准确率为0
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
        else:
            accuracy = self.correct_predictions / (self.total_predictions - self.invalid_responses) * 100
        # 打开输出文件，以写入模式保存结果
        with open(output_path, 'w') as f:
            # 准备要保存的结果数据
            results_to_save = {
                'inference_prompt': self.inference_prompt,  # 推理提示
                'prompt': self.gpt_prompt,  # GPT提示
                'accuracy': f"{accuracy:.2f}%",  # 准确率，格式化为两位小数
                'total_predictions': self.total_predictions,  # 总预测数
                'correct_predictions': self.correct_predictions,  # 正确预测数
                'invalid_responses': self.invalid_responses,  # 无效响应数
                'prompt_tokens': self.prompt_tokens,  # 提示令牌数
                'completion_tokens': self.completion_tokens,  # 完成令牌数
                'GPT_cost': self.get_costs(),  # GPT成本
                'results': self.response_data,  # 响应数据
            }
            # 将结果数据以JSON格式写入文件，缩进为2个空格
            json.dump(results_to_save, f, indent=2)
        
        # 打印保存结果的路径
        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")
    
    def print_results(self):
        # 打印分隔线，用于美化输出
        print('-' * 80)
        # 检查总预测数减去无效响应数是否为0，以避免除零错误
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
        else:
            accuracy = self.correct_predictions / (self.total_predictions - self.invalid_responses) * 100
        print("Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Correct Predictions: {self.correct_predictions}")
        print(f"Invalid Responses: {self.invalid_responses}")
        self.print_costs()
    
    def print_costs(self):
        # 计算并打印提示令牌的总价格
        # self.prompt_tokens 是提示令牌的数量
        # self.price_1k_prompt_tokens 是每千个提示令牌的价格
        # 将总价格除以1000并格式化为两位小数，然后打印出来
        print(f"Prompt Tokens Price: {self.prompt_tokens * self.price_1k_prompt_tokens / 1000:.2f} USD")
        print(f"Completion Tokens Price: {self.completion_tokens * self.price_1k_completion_tokens / 1000:.2f} USD")
    
    def get_costs(self):
        return self.prompt_tokens * self.price_1k_prompt_tokens / 1000 + self.completion_tokens * self.price_1k_completion_tokens / 1000


class OpenAICloseSetClsEvaluator(OpenAIOpenFreeFormClsEvaluator):
    def __init__(self, inputs, output_dir, output_file, model_type="gpt-3.5-turbo-0613"):
        # 调用父类的构造函数
        super().__init__(inputs, output_dir, output_file, model_type)
        # 根据模型类型选择不同的提示模板
        self.gpt_prompt = chatgpt_close_set_cls_prompt if "gpt-3.5" in model_type else gpt4_close_set_cls_prompt

        self.invalid_correct_predictions = 0 # * random choice and correct coincidently

        # * import category names
        try:
            # * load a txt files of category names
            catfile = os.path.join(os.path.dirname(__file__), '../data/modelnet_config/modelnet40_shape_names_modified.txt') # * i.e. pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt
            self.candidate_lists_names = [line.strip() for line in open(catfile)] # * list of category names
        except:
            print(f"Current categories file is {catfile}. Need to move the category file to pointllm/eval/configs/.") 

        # * make the prompt
        candidate_lists = [f'{i}: {cat}' for i, cat in enumerate(self.candidate_lists_names)]
        self.num_categories = len(candidate_lists)
        self.candidate_lists = '\n'.join(candidate_lists)
        self.gpt_prompt = self.gpt_prompt.format(num_categories=self.num_categories, candidate_lists=self.candidate_lists) + "{model_output}\nOutput: "
    
    def check_model_type(self):
        # * no need to check for this task
        # 这行注释表明当前函数对于当前任务来说不需要执行任何检查操作
        return
    
    def resume_processing(self):
        # 构建已处理结果的文件路径
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        # 检查文件是否存在
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.correct_predictions = saved_results["correct_predictions"]
            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.invalid_correct_predictions = saved_results["invalid_correct_predictions"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")

    def parse_gpt_response_evaluate(self, gpt_response, ground_truth):
        """
        解析GPT模型的响应并评估其准确性。
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """

        # * use regular expression to extract
        pattern = r'(\d+#[^#]*#.*$)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        cls_result = gpt_response_list[0]
        cls_label = gpt_response_list[1] if len(gpt_response_list) > 1 else ""
        reason = gpt_response_list[2] if len(gpt_response_list) > 2 else ""

        try:
            # * convert to int
            cls_result = int(cls_result)
            if cls_result not in range(self.num_categories) or cls_label == "NA":
                # * not valid range
                cls_result = -1
        except ValueError:
            print(f"Error: unale to parse {gpt_response}.")
            cls_result = -1

        if cls_result == -1:
            # * random choose one index from 0 to self.num_categories
            cls_result = random.choice(range(self.num_categories))
            cls_label = "INVALID"
            reason = gpt_response

            self.invalid_responses += 1
        
        accuracy = 1 if cls_result == ground_truth else 0 

        return accuracy, cls_result, cls_label, reason

    def evaluate_result(self, result):
        # 从结果字典中获取object_id，如果不存在则默认为-1
        object_id = result.get('object_id', -1)
        # 获取ground_truth，即真实标签
        ground_truth = result['ground_truth']
        # 获取标签名称
        ground_truth_label = result['label_name']
        # 获取模型输出
        model_output = result['model_output']

        # 构造与GPT交互的消息，用户角色，内容为格式化后的提示信息
        messages = [{"role": "user", "content": self.gpt_prompt.format(model_output=model_output)}]
        
        # 使用openaigpt的safe_chat_complete方法与GPT进行交互，获取响应
        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 

        # 从GPT响应中获取提示的token数量
        prompt_tokens = gpt_response['usage']['prompt_tokens']
        # 从GPT响应中获取完成的token数量
        completion_tokens = gpt_response['usage']['completion_tokens']

        # 从GPT响应中获取实际的内容
        gpt_response = gpt_response['choices'][0]["message"]['content']

        accuracy, cls_result, cls_label, reason = self.parse_gpt_response_evaluate(gpt_response, ground_truth) # return 0, "INVALID", gpt_response if not valid

        return object_id, model_output, ground_truth, accuracy, cls_result, cls_label, reason, ground_truth_label, prompt_tokens, completion_tokens

    def evaluate(self):

        # 调用resume_processing方法，可能是用于恢复或初始化某些处理状态
        self.resume_processing()
        
        # 打印分隔线
        print('-' * 80)
        # 打印开始单线程评估的提示信息
        print("Starting single-thread evaluation...")
        # 获取评估结果
        results = self.results

        try:
            # 使用tqdm库对results进行迭代，显示进度条
            for result in tqdm(results):  
                # 调用evaluate_result方法对每个结果进行评估，返回多个参数
                object_id, model_output, ground_truth, accuracy, cls_result, cls_label, reason, ground_truth_label, prompt_tokens, completion_tokens = self.evaluate_result(result)
                # 累加正确的预测数量
                self.correct_predictions += accuracy
                # 累加总的预测数量
                self.total_predictions += 1
                
                # 如果分类标签为"INVALID"，则分别累加无效的正确预测数量和无效响应数量
                if cls_label == "INVALID":
                    self.invalid_correct_predictions += accuracy
                    self.invalid_responses += 1
                
                # 累加提示和完成令牌数量
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'gpt_cls_result': cls_result,
                    'ground_truth_label': ground_truth_label,
                    'gpt_cls_label': cls_label,
                    'model_output': model_output,
                    'gpt_reason': reason,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                })
            
            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            print(f"Current sample is {result}.")
            self.save_results(is_temp=True)
            exit()
    
    def parallel_evaluate(self, num_workers=20):

        # 定义一个方法用于并行评估，num_workers参数指定并行工作的数量，默认为20
        self.resume_processing()
        
        # 调用resume_processing方法，可能是用于恢复之前的处理状态
        print('-' * 80)
        # 打印80个连字符，用于分隔输出
        print("Starting parallel evaluation...")
        # 打印提示信息，表示开始并行评估
        results = self.results

        # 获取评估结果，存储在results变量中
        try:
            # 使用try-except结构来捕获可能出现的异常
            with Pool(num_workers) as pool:
                # 创建一个进程池，指定并行工作的数量
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    # 使用tqdm创建一个进度条，total参数指定进度条的总长度为results的长度
                    for object_id, model_output, ground_truth, accuracy, cls_result, cls_label, reason, ground_truth_label, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        # 使用imap_unordered方法并行地调用evaluate_result方法处理results中的每个元素
                        # evaluate_result方法的返回值被解包到多个变量中
                        self.correct_predictions += accuracy
                        # 将准确度累加到correct_predictions中
                        self.total_predictions += 1

                        # 总预测数加1
                        self.prompt_tokens += prompt_tokens
                        # 将prompt_tokens累加到self.prompt_tokens中
                        self.completion_tokens += completion_tokens

                        # 将completion_tokens累加到self.completion_tokens中
                        if cls_label == "INVALID":
                            # 如果分类标签为"INVALID"
                            self.invalid_correct_predictions += accuracy
                            # 将准确度累加到invalid_correct_predictions中
                            self.invalid_responses += 1

                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'gpt_cls_result': cls_result,
                            'ground_truth_label': ground_truth_label,
                            'gpt_cls_label': cls_label,
                            'model_output': model_output,
                            'gpt_reason': reason,
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens
                        })

                        pbar.update()  # update the progress bar

            print("Parallel evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()

        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit() 

    def save_results(self, is_temp=False):
        # 根据是否为临时保存，确定输出文件路径
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        # 如果总预测数减去无效响应数为0，则准确率和清洗后准确率都设为0
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
            clean_accuracy = 0
        else:
            # 计算准确率：正确预测数除以总预测数，乘以100
            accuracy = self.correct_predictions / self.total_predictions * 100
            # 计算清洗后准确率：有效正确预测数除以有效总预测数，乘以100
            clean_accuracy = (self.correct_predictions - self.invalid_correct_predictions) / (self.total_predictions - self.invalid_responses) * 100
        # 打开输出文件，以写入模式
        with open(output_path, 'w') as f:
            # 准备要保存的结果数据
            results_to_save = {
                'inference_prompt': self.inference_prompt,  # 推理提示
                'prompt': self.gpt_prompt,  # GPT提示
                'accuracy': f"{accuracy:.2f}%",  # 准确率，保留两位小数
                'clean_accuracy': f"{clean_accuracy:.2f}%",  # 清洗后准确率，保留两位小数
                'total_predictions': self.total_predictions,  # 总预测数
                'correct_predictions': self.correct_predictions,  # 正确预测数
                'invalid_correct_predictions': self.invalid_correct_predictions,  # 无效的正确预测数
                'invalid_responses': self.invalid_responses,  # 无效响应数
                'prompt_tokens': self.prompt_tokens,  # 提示令牌数
                'completion_tokens': self.completion_tokens,  # 完成令牌数
                'GPT_cost': self.get_costs(),   # GPT成本
                'results': self.response_data,  # 响应数据
            }
            # 将结果数据以JSON格式写入文件，缩进为2个空格
            json.dump(results_to_save, f, indent=2)
        
        # 打印保存结果的路径
        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")
    
    def print_results(self):
        # 打印分隔线
        print('-' * 80)
        # 检查总预测数减去无效响应数是否为0，如果是则准确率为0，避免除以0错误
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
        else:
            # 计算准确率：正确预测数除以总预测数，乘以100得到百分比
            accuracy = self.correct_predictions / self.total_predictions * 100
            # 计算干净准确率：去除无效正确预测后的正确预测数除以去除无效响应后的总预测数，乘以100得到百分比
            clean_accuracy = (self.correct_predictions - self.invalid_correct_predictions) / (self.total_predictions - self.invalid_responses) * 100
        # 重新计算准确率（此处重复计算，可能为代码逻辑错误）
        accuracy = self.correct_predictions / self.total_predictions * 100
        # 打印结果信息
        print("Results:")
        print(f"Accuracy: {accuracy:.2f}%")  # 打印准确率，保留两位小数
        print(f"Clean Accuracy: {clean_accuracy:.2f}%",)  # 打印干净准确率，保留两位小数
        print(f"Total Predictions: {self.total_predictions}")  # 打印总预测数
        print(f"Correct Predictions: {self.correct_predictions}")  # 打印正确预测数
        print(f"Invalid Correct Predictions: {self.invalid_correct_predictions}")  # 打印无效正确预测数
        print(f"Invalid Responses: {self.invalid_responses}")  # 打印无效响应数
        print(f"Prompt Tokens: {self.prompt_tokens}")  # 打印提示令牌数
        print(f"Completion Tokens: {self.completion_tokens}")  # 打印完成令牌数

        # 调用打印成本的函数
        self.print_costs()
    
class OpenAIObjectCaptioningEvaluator(OpenAIOpenFreeFormClsEvaluator):
    def __init__(self, inputs, output_dir, output_file, model_type="gpt-4-0613"):
        super().__init__(inputs, output_dir, output_file, model_type)
        self.gpt_prompt = chatgpt_object_captioning_prompt if "gpt-3.5" in model_type else gpt4_object_captioning_prompt

        self.total_scores = 0

    def resume_processing(self):
        # 构建已处理结果的文件路径
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        # 检查文件是否存在
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.total_scores = float(saved_results["total_score"])

            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")

    def parse_gpt_response_evaluate(self, gpt_response, ground_truth):
        """
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """

        # * use regular expression to extract
        pattern = r'(\d*#.*)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        gpt_score = gpt_response_list[0]
        reason = gpt_response_list[1] if len(gpt_response_list) > 1 else ""

        try:
            # * convert to int
            gpt_score = int(gpt_score)
            if gpt_score not in range(101): # * in 0-100
                # * not valid range
                gpt_score = -1
        except ValueError:
            print(f"Error: unale to parse {gpt_response}.")
            gpt_score = -1

        if gpt_score == -1:
            reason = gpt_response
        
        return gpt_score, reason

    def evaluate_result(self, result):
        # 从结果字典中获取对象ID，如果不存在则默认为-1
        object_id = result.get('object_id', -1)
        # 获取真实标签（ground truth）
        ground_truth = result['ground_truth']
        # 获取模型输出
        model_output = result['model_output']

        # 构造与GPT交互的消息，用户角色内容为格式化后的提示信息
        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]
        
        # 使用OpenAI GPT安全地完成聊天，获取GPT的响应
        gpt_response = self.openaigpt.safe_chat_complete(messages, content_only=False) 

        # 从GPT响应中获取提示令牌数和完成令牌数
        prompt_tokens = gpt_response['usage']['prompt_tokens']
        completion_tokens = gpt_response['usage']['completion_tokens']

        # 从GPT响应中提取GPT生成的消息内容
        gpt_response = gpt_response['choices'][0]["message"]['content']

        gpt_score, reason = self.parse_gpt_response_evaluate(gpt_response, ground_truth) # return 0, "INVALID", gpt_response if not valid

        return object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens

    def evaluate(self):

        # 调用resume_processing方法，可能是为了恢复之前暂停的处理
        self.resume_processing()
        
        # 打印分隔线
        print('-' * 80)
        # 打印开始单线程评估的提示信息
        print("Starting single-thread evaluation...")
        # 获取结果列表
        results = self.results

        try:
            # 使用tqdm遍历结果列表，tqdm用于显示进度条
            for result in tqdm(results):  
                # 调用evaluate_result方法对每个结果进行评估，返回多个值
                object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens = self.evaluate_result(result)

                # 累加有效的gpt_score到total_scores，如果gpt_score为-1则不累加
                self.total_scores += gpt_score if gpt_score != -1 else 0
                # 累加预测总数
                self.total_predictions += 1
                # 累加提示词数
                self.prompt_tokens += prompt_tokens
                # 累加完成词数
                self.completion_tokens += completion_tokens
                
                # 如果gpt_score为-1，表示无效响应，累加无效响应数
                if gpt_score == -1:
                    self.invalid_responses += 1

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'model_output': model_output,
                    "gpt_score": gpt_score,
                    'gpt_reason': reason
                })
            
            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()
    
    def parallel_evaluate(self, num_workers=20):

        # 定义一个方法用于并行评估，默认使用20个工作进程
        self.resume_processing()
        
        # 调用resume_processing方法，可能用于恢复之前的处理状态
        print('-' * 80)
        # 打印80个连字符，用于分隔输出
        print("Starting parallel evaluation...")
        # 打印开始并行评估的信息
        results = self.results

        # 获取评估结果列表
        try:
            # 使用try-except结构来捕获可能的异常
            with Pool(num_workers) as pool:
                # 创建一个进程池，指定工作进程数量
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    # 使用tqdm创建一个进度条，总长度为结果列表的长度
                    for object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        # 使用imap_unordered方法并行地调用evaluate_result函数，处理每个结果
                        # evaluate_result函数返回多个值：object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens
                        self.total_scores += gpt_score if gpt_score != -1 else 0
                        # 累加有效的gpt_score到total_scores，无效的（-1）不计入
                        self.total_predictions += 1
                        # 累加预测总数
                        self.prompt_tokens += prompt_tokens
                        # 累加提示令牌数
                        self.completion_tokens += completion_tokens
                        
                        # 累加完成令牌数
                        if gpt_score == -1:
                            self.invalid_responses += 1

                        # 如果gpt_score为-1，表示无效响应，累加无效响应数
                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            "gpt_score": gpt_score,
                            'gpt_reason': reason
                        })

                        # 将每个结果的相关数据保存到response_data列表中
                        pbar.update()  # update the progress bar

                        # 更新进度条
            print("Parallel evaluation finished.")

            # 打印并行评估完成的信息
            self.save_results()
            # 调用save_results方法保存评估结果
            self.print_results()
            # 调用print_results方法打印评估结果
            self.remove_temp_file()

            # 调用remove_temp_file方法移除临时文件
        except (Exception, KeyboardInterrupt) as e:
            # 捕获异常或键盘中断
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            # 打印错误信息，并提示将已处理的结果保存到临时文件
            self.save_results(is_temp=True)
            # 调用save_results方法，传入is_temp=True参数，保存为临时文件
            exit() 

    def save_results(self, is_temp=False):
        # 如果is_temp为True，则使用临时输出文件名，否则使用正式输出文件名
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        # 如果总预测数减去无效响应数为0，则平均分为0，避免除以0错误
        if self.total_predictions - self.invalid_responses == 0:
            average_score = 0 # * no results and get error
        else:
            # 计算平均分，总分数除以有效预测数
            average_score = self.total_scores / (self.total_predictions - self.invalid_responses)
        # 打开输出文件，以写入模式
        with open(output_path, 'w') as f:
            # 准备要保存的结果数据
            results_to_save = {
                'inference_prompt': self.inference_prompt,  # 推理提示
                'gpt_prompt': self.gpt_prompt,              # GPT提示
                'average_score': f"{average_score:.2f}",    # 平均分，保留两位小数
                'total_score': f"{self.total_scores:.2f}",  # 总分数，保留两位小数
                'total_predictions': self.total_predictions, # 总预测数
                'invalid_responses': self.invalid_responses, # 无效响应数
                'prompt_tokens': self.prompt_tokens,        # 提示令牌数
                'completion_tokens': self.completion_tokens, # 完成令牌数
                'GPT_cost': self.get_costs(),                 # GPT成本
                'results': self.response_data,               # 响应数据
            }
            # 将结果数据以JSON格式写入文件，缩进为2个空格
            json.dump(results_to_save, f, indent=2)
        
        # 打印保存结果的路径
        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")
    
    def print_results(self):
        # 打印分隔线，用于美化输出
        print('-' * 80)
        # 检查总预测数减去无效响应数是否为0，避免除以0的错误
        if self.total_predictions - self.invalid_responses == 0:
            average_score = 0 # * no results and get error
        else:
            average_score = self.total_scores / (self.total_predictions - self.invalid_responses)
        print("Results:")
        print(f"Average Score: {average_score:.2f}")
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Invalid Responses: {self.invalid_responses}")
        print(f"Prompt Tokens: {self.prompt_tokens}")
        print(f"Completion Tokens: {self.completion_tokens}")

        self.print_costs()


def start_evaluation(results, output_dir, output_file, eval_type="open-free-form-classification", model_type="gpt-3.5-turbo-0613",
                        parallel=True, num_workers=20):
    """
    开始评估函数，根据不同的评估类型和模型类型对结果进行评估，并将评估结果保存到指定文件。
    Args:
        results: dict or file path to the json file containing the dict
        output_file: the path the final evaluation results to be saved.
    """
    if isinstance(results, str):
        with open(results, 'r') as fp:
            results = json.load(fp)

    if eval_type == "open-free-form-classification":
        evaluator = OpenAIOpenFreeFormClsEvaluator(results, output_dir, output_file, model_type=model_type)
    elif eval_type == "modelnet-close-set-classification":
        evaluator = OpenAICloseSetClsEvaluator(results, output_dir, output_file, model_type=model_type)
    elif eval_type == "object-captioning":
        evaluator = OpenAIObjectCaptioningEvaluator(results, output_dir, output_file, model_type=model_type)
    else:
        raise NotImplementedError(f"eval_type {eval_type} not supported.")

    if parallel:
        evaluator.parallel_evaluate(num_workers=num_workers)
    else:
        evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, \
                        default="", help="Path to the results file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")
    parser.add_argument("--model_type", type=str, default="gpt-4-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--parallel", default=True, action="store_true", help="Whether to use parallel evaluation.")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers to use for parallel evaluation.")
    parser.add_argument("--eval_type", type=str, choices=["modelnet-close-set-classification", "open-free-form-classification", "object-captioning"], default="object-captioning")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_path)

    output_file = os.path.basename(args.results_path).replace(".json", f"_evaluated_{args.model_type}.json")

    # if exists, then exit
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        print(f"[INFO] Evaulated results already exists in {os.path.join(args.output_dir, output_file)}.")
        exit()

    start_evaluation(results=args.results_path, output_dir=args.output_dir, output_file=output_file, eval_type=args.eval_type, model_type=args.model_type, 
                        parallel=args.parallel, num_workers=args.num_workers)
    