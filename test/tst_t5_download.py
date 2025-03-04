import os

# 设置环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:50201'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:50201'

from huggingface_hub import login
login("your_token")
 
from transformers import T5EncoderModel, T5Tokenizer
import requests

# 指定预训练模型的版本
version = "xlabs-ai/xflux_text_encoders"  # 或者 "t5-base", "t5-large" 等

# 设置代理配置
proxies = {
    'http': 'http://127.0.0.1:50201',
    'https': 'http://127.0.0.1:50201'
}


# 测试连接
try:
    # 测试 Google
    response1 = requests.get('https://www.google.com', proxies=proxies, timeout=5)
    print("Google 连接状态:", response1.status_code)
    
    # 测试 Hugging Face
    response2 = requests.get('https://huggingface.co', proxies=proxies, timeout=5)
    print("HuggingFace 连接状态:", response2.status_code)
except Exception as e:
    print("连接失败:", str(e))

# 打印当前环境信息
import sys
print("\nPython 版本:", sys.version)
print("Requests 版本:", requests.__version__)

# 配置请求session
session = requests.Session()
session.proxies.update(proxies)

# 更新 hf_kwargs
hf_kwargs = {
    "proxies": proxies,
    "timeout": 1000,  # 增加超时时间
    "local_files_only": False,
    "trust_remote_code": True,
    "use_auth_token": True
}

# 加载 T5 编码器模型
model = T5EncoderModel.from_pretrained(version, **hf_kwargs)

# 加载对应的 tokenizer
tokenizer = T5Tokenizer.from_pretrained(version)

# 示例：使用模型对文本进行编码
text = "这是一个示例文本。"
inputs = tokenizer(text, return_tensors="pt")  # 将文本转换为模型输入

# 获取编码器的输出
outputs = model(**inputs)

# 输出结果
print(outputs.last_hidden_state)  # 输出最后一个隐藏状态

