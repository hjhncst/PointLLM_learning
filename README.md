# data文件夹

## data generation文件夹

### system_prompt_gpt4_0613.txt

&emsp;&emsp;指导 GPT-4 生成复杂指令的系统提示文件，主要用yu数据生成流程，帮助生成与 3D 对象相关的详细描述和问答对话。具体来说，文件要求模型执行以下任务：

1.生成详细描述：从给定的 3D 对象模型描述中，生成一段 50 至 100 个单词的详细说明，描述对象的类型、外观、功能及其在日常生活中的应用，但排除不确定的细节。

2.单轮问答生成：基于生成的描述，生成三个单轮问答对话，每个问答关注对象的不同方面。

3.多轮问答生成：构造一组包含三轮问答的对话，要求问答之间逻辑相关且内容不同于单轮问答部分。

## modelnet_config文件夹

### modelnet40_shape_names_modified.txt

&emsp;&emsp;包含 ModelNet40 数据集中所有对象的名称，用于在训练过程中生成对象名称的描述。

### ModelNet40.yaml

&emsp;&emsp;配置文件，定义了 ModelNet40 数据集的参数，包括数据路径、对象名称文件路径、训练集和测试集的划分等。

## modelnet.py

<p align="center">
  <img src="assets/image1.png" align="center" width="100%">
</p>

根据索引加载单个数据样本，并对其做预处理。

<p align="center">
  <img src="assets/image2.png" align="center" width="100%">
</p>

<p align="center">
  <img src="assets/image3.png" align="center" width="100%">
</p>

对输入点云进行归一化，确保其中心化并缩放到单位范围内。

<p align="center">
  <img src="assets/image4.png" align="center" width="100%">
</p>

从数据集中取出一个样本，并将其转换为 PyTorch tensor，同时构造包含额外信息的字典返回。

## object_point_dataset.py

