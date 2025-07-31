# llm.py
import aiohttp
import asyncio
import re
import logging
import json

# 简化注释：获取日志记录器
logger = logging.getLogger(__name__)

class LLMClient:
    """
    简化注释：用于与大语言模型（如Ollama）交互的客户端，支持流式响应。
    """
    def __init__(self, url, model, system_prompt):
        """
        简化注释：初始化LLM客户端。
        - url: 模型服务的API URL。
        - model: 使用的模型名称。
        - system_prompt: 发送给模型的系统级提示。
        """
        self.url = url
        self.model = model
        self.system_prompt = system_prompt
        self.history = []

    async def _call_raw_stream(self, prompt: str, timeout=60):
        """
        简化注释：向LLM服务发送流式请求并以异步生成器方式返回内容块。
        - prompt: 用户输入的提示。
        - timeout: 请求超时时间（秒）。
        - 返回: LLM响应内容的异步生成器。
        """
        self.history.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "stream": True,  # 关键修复：启用流式响应
            "messages": [
                {"role": "system", "content": self.system_prompt}
            ] + self.history[-10:] # 保留最近10条历史记录
        }

        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.url, json=payload, timeout=timeout) as r:
                async for line in r.content:
                    if line:
                        try:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line:
                                chunk = json.loads(decoded_line)
                                if chunk.get("done") == False:
                                     yield chunk.get("message", {}).get("content", "")
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"Failed to decode line: {line}, error: {e}")
                            continue
    
    @staticmethod
    def _clean(text: str) -> str:
        """
        简化注释：清洗LLM返回的文本，移除思考标签和多余标点。
        - text: 原始文本。
        - 返回: 清洗后的文本。
        """
        # ① 删除 <think>…</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        # ② 删除 /nothink 等残留
        text = re.sub(r"/nothink", "", text, flags=re.I)
        # ③ 移除除中文、字母、数字、空格、逗号、句号外的所有符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。.,]', '', text)
        return text.strip()

    async def ask(self, prompt: str):
        """
        简化注释：以流式方式向LLM提问，并异步生成干净的、可播报的答案片段。
        - prompt: 用户输入的提示。
        - 返回: 处理过的LLM响应文本片段的异步生成器。
        """
        try:
            buffer = ""
            async for raw_chunk in self._call_raw_stream(prompt):
                if raw_chunk:
                    buffer += raw_chunk
                    # 按句子结束符或换行符分割，确保推送完整的句子或段落
                    while any(p in buffer for p in ['。', '！', '？', '...', '”', '\n']):
                        # 找到第一个结束符的位置
                        split_points = [buffer.find(p) for p in ['。', '！', '？', '...', '”', '\n'] if p in buffer]
                        first_split_point = min(split_points) + 1
                        
                        sentence = buffer[:first_split_point]
                        buffer = buffer[first_split_point:]
                        
                        cleaned_sentence = self._clean(sentence)
                        if cleaned_sentence:
                            yield cleaned_sentence

            # 处理最后一个不含结束符的句子
            if buffer:
                cleaned_buffer = self._clean(buffer)
                if cleaned_buffer:
                    yield cleaned_buffer + "……" # 用省略号表示对话可能未完全结束
                
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            yield "抱歉，我暂时无法回答。"

# 保持 ask_llm 的同步封装，但需注意，它将把流式响应聚合为单个字符串，从而失去流式处理的优势。
# 在需要真正实时响应的地方，应直接调用异步的 ask 方法。
def ask_llm(prompt: str) -> str:
    """
    简化注释：为非异步代码提供一个同步的LLM调用封装。
    注意：此函数会聚合所有流式响应，表现为阻塞行为。
    - prompt: 用户输入的提示。
    - 返回: 处理过的LLM完整响应。
    """
    async def _collect_stream():
        response_parts = []
        async for part in llm_client.ask(prompt):
            response_parts.append(part)
        return "".join(response_parts)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            task = loop.create_task(_collect_stream())
            # 这是一个简单的同步等待实现，不建议在复杂的异步应用中使用
            while not task.done():
                loop.run_until_complete(asyncio.sleep(0.1))
            return task.result()
        else:
            return asyncio.run(_collect_stream())
    except RuntimeError:
        return asyncio.run(_collect_stream())