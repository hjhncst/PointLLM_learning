import argparse
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from tqdm import tqdm
from pointllm.eval.evaluator import start_evaluation
from transformers import AutoTokenizer

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

# 初始化模型
def init_model(args):
    # Model
    disable_torch_init()  # 禁用PyTorch的初始化，以提高加载模型的效率
    # 获取模型名称，并展开用户路径（例如，将~替换为用户主目录）
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

# 加载数据集
def load_dataset(config_path, split, subset_nums, use_color):
    # 打印信息，表示正在加载指定数据集的split部分
    print(f"Loading {split} split of ModelNet datasets.")
    # 创建ModelNet数据集对象，传入配置路径、数据集部分、子集数量和使用颜色等信息
    dataset = ModelNet(config_path=config_path, split=split, subset_nums=subset_nums, use_color=use_color)
    # 打印信息，表示数据集加载完成
    print("Done!")
    # 返回加载的数据集对象
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    # 断言shuffle为False，因为在评估时我们使用ModelNet的索引作为对象ID，
    # 因此shuffle应该为False，并且应该始终设置随机种子。
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    # 创建一个数据加载器，传入数据集、批量大小、是否打乱数据以及工作线程数。
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 返回创建好的数据加载器。
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    # 将模型设置为评估模式，这样在进行推理时不会更新模型参数
    model.eval() 
    # 使用torch.inference_mode上下文管理器，以减少内存使用并提高推理速度
    with torch.inference_mode():
        # 使用模型的generate方法生成输出序列
        output_ids = model.generate(
            input_ids,  # 输入的token ID序列
            point_clouds=point_clouds,  # 点云数据
            do_sample=do_sample,  # 是否进行采样生成
            temperature=temperature,  # 控制生成文本的随机性
            top_k=top_k,  # 限制生成时考虑的top k个概率最高的token
            max_length=max_length,  # 输出序列的最大长度
            top_p=top_p,  # 用于核采样（nucleus sampling）的阈值
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file):
    # 获取停止字符串，如果分隔符样式不是TWO，则使用conv.sep，否则使用conv.sep2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # 从PROMPT_LISTS中获取指定索引的提示
    qs = PROMPT_LISTS[prompt_index]

    # 初始化结果字典，包含提示
    results = {"prompt": qs}

    # 获取模型的point_backbone配置
    point_backbone_config = model.get_model().point_backbone_config
    # 获取point_token的长度
    point_token_len = point_backbone_config['point_token_len']
    # 获取默认的point_patch_token
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    # 获取默认的point_start_token
    default_point_start_token = point_backbone_config['default_point_start_token']
    # 获取默认的point_end_token
    default_point_end_token = point_backbone_config['default_point_end_token']
    # 获取是否使用point_start和point_end的标志
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    # 根据标志决定如何构造输入字符串
    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    # 将构造好的输入字符串添加到对话中
    conv.append_message(conv.roles[0], qs)
    # 添加一个空的回复
    conv.append_message(conv.roles[1], None)

    # 获取对话的完整提示
    prompt = conv.get_prompt()
    # 使用tokenizer对提示进行编码
    inputs = tokenizer([prompt])

    # 将输入ID转换为CUDA张量
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    # 创建关键词停止标准
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    # 初始化响应列表
    responses = []

    # 遍历数据加载器中的每个批次
    for batch in tqdm(dataloader):
        # 获取点云数据并转换为CUDA张量
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        # 获取标签
        labels = batch["labels"]
        # 获取标签名称
        label_names = batch["label_names"]
        # 获取索引
        indice = batch["indice"]

        # 获取批次大小
        batchsize = point_clouds.shape[0]

        # 将输入ID重复为批次大小
        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        # 生成输出
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        # saving results
        for index, output, label, label_name in zip(indice, outputs, labels, label_names):
            responses.append({
                "object_id": index.item(),
                "ground_truth": label.item(),
                "model_output": output,
                "label_name": label_name
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")

    # * output file 
    args.output_file = f"ModelNet_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need to generate results first
        dataset = load_dataset(config_path=None, split=args.split, subset_nums=args.subset_nums, use_color=args.use_color) # * defalut config
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
    
        model, tokenizer, conv = init_model(args)

        # * ouptut
        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    # * evaluation file
    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    # * start evaluation
    if args.start_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type="modelnet-close-set-classification", model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2")

    # * dataset type
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--subset_nums", type=int, default=-1) # * only use "subset_nums" of samples, mainly for debug 

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")

    args = parser.parse_args()

    main(args)
