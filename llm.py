# llm.py
import aiohttp
import asyncio
import re
import logging

# 简化注释：获取日志记录器
logger = logging.getLogger(__name__)

class LLMClient:
    """
    简化注释：用于与大语言模型（如Ollama）交互的客户端。
    """
    def __init__(self, url, model, system_prompt):
        """
        简化注释：初始化LLM客户端。
        - url: 模型服务的API URL。
        - model: 使用的模型名称。
        - system_prompt: 发送给模型的系统级提示。
        """
        self.url, self.model, self.system_prompt = url, model, system_prompt

    async def _call_raw(self, prompt: str, timeout=60) -> str:
        """
        简化注释：向LLM服务发送原始请求并获取响应。
        - prompt: 用户输入的提示。
        - timeout: 请求超时时间（秒）。
        - 返回: LLM的原始文本响应。
        """
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt}
            ]
        }
        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.url, json=payload, timeout=timeout) as r:
                data = await r.json()
                return data.get("message", {}).get("content", "")

    @staticmethod
    def _clean(text: str) -> str:
        """
        简化注释：清洗LLM返回的文本，移除思考标签。
        - text: 原始文本。
        - 返回: 清洗后的文本。
        """
        # ① 删除 <think>…</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        # ② 删除 /nothink 等残留
        text = re.sub(r"/nothink", "", text, flags=re.I)
        return text.strip()

    async def ask(self, prompt: str) -> str:
        """
        简化注释：向LLM提问并返回干净的、可播报的答案。
        - prompt: 用户输入的提示。
        - 返回: 处理过的LLM响应或错误信息。
        """
        try:
            raw = await self._call_raw(prompt)
            # 简化注释：如果清洗后为空，则返回省略号
            return self._clean(raw) or "……"
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return "抱歉，我暂时无法回答。"

def ask_llm(prompt: str) -> str:
    """
    简化注释：为非异步代码提供一个同步的LLM调用封装。
    - prompt: 用户输入的提示。
    - 返回: 处理过的LLM响应。
    """
    # 简化注释：在已有事件循环的情况下不能直接运行asyncio.run()
    # 这里的实现仅为示例，实际应用中需根据主程序事件循环情况调整
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # 简化注释：如果事件循环正在运行，则创建任务
            task = loop.create_task(llm_client.ask(prompt))
            # 简化注释：注意：此处的同步等待会阻塞当前异步函数，仅用于简单示例
            # 实际应用中应避免在异步函数中这样调用
            return asyncio.run(asyncio.sleep(0, task.result()))
        else:
            return asyncio.run(llm_client.ask(prompt))
    except RuntimeError:
        # 简化注释：如果没有正在运行的事件循环，则启动一个新的
        return asyncio.run(llm_client.ask(prompt))