import openai
import time
import random
import os

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout),
):
    """Retry a function with exponential backoff."""
    # 定义一个内部函数wrapper，用于包装传入的函数func
    def wrapper(*args, **kwargs):
        num_retries = 0  # 初始化重试次数为0
        delay = initial_delay  # 初始化延迟时间为initial_delay

        while True:
            try:
                # 尝试执行传入的函数func，并返回其结果
                return func(*args, **kwargs)
            except errors as e:
                # * print the error info
                num_retries += 1
                if num_retries > max_retries:
                    print(f"[OPENAI] Encounter error: {e}.")
                    raise Exception(
                        f"[OPENAI] Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                raise e
    return wrapper

class OpenAIGPT():
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=1, top_p=1, max_tokens=2048, **kwargs) -> None:

        # 初始化OpenAIGPT类，设置默认的聊天参数
        setup_openai(model)  # 调用setup_openai函数，传入模型名称进行初始化
        self.default_chat_parameters = {
            "model": model,   # 设置默认模型
            "temperature": temperature,   # 设置默认温度参数，控制生成文本的多样性
            "top_p": top_p,   # 设置默认top_p参数，控制生成文本的随机性
            "max_tokens": max_tokens,  # 设置默认最大令牌数，限制生成文本的长度
            **kwargs  # 将其他可选参数添加到默认参数中
        }

    @retry_with_exponential_backoff 
    def safe_chat_complete(self, messages, content_only=True, **kwargs):

        # 使用指数退避重试机制装饰的safe_chat_complete方法，用于安全地完成聊天
        chat_parameters = self.default_chat_parameters.copy()  # 复制默认聊天参数
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)  # 如果有额外的参数，更新到聊天参数中

        response = openai.ChatCompletion.create(
            messages=messages,  # 传入聊天消息
            **chat_parameters  # 传入聊天参数
        )

        if content_only:
            # 如果content_only为True，只返回生成文本的内容
            response = response['choices'][0]["message"]['content']

        return response  # 返回最终的响应结果

def setup_openai(model_name):
    # Setup OpenAI API Key
    print("[OPENAI] Setting OpenAI api_key...")
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(f"[OPENAI] OpenAI organization: {openai.organization}")
    print(f"[OPENAI] Using MODEL: {model_name}")