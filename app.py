# app.py
# 导入必要的库
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
from functools import partial # 简化注释：导入 partial 用于包装函数
import re
import numpy as np
from threading import Thread,Event
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
import gc

# 简化注释：从 funasr 和 llm 模块导入必要的类和函数
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile
import os
from llm import LLMClient


app = Flask(__name__)
nerfreals:Dict[int, BaseReal] = {} # 简化注释：存储会话ID与虚拟人实例的映射
opt = None
model = None
avatar = None
asr_model = None # 简化注释：ASR模型实例
llm_client = None # 简化注释：LLM客户端实例


##### webrtc ###############################
pcs = set()

def randN(N)->int:
    '''简化注释：生成一个指定长度的随机数'''
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    """简化注释：根据配置构建并返回一个虚拟人实例"""
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
        pass # ernerf 模型暂未实现
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

async def offer(request):
    """简化注释：处理WebRTC的offer请求，建立P2P连接"""
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    nerfreals[sessionid] = None
    logger.info('创建会话 sessionid=%d, 当前会话数=%d',sessionid,len(nerfreals))

    ice_servers = [RTCIceServer(urls="stun:stun.miwifi.com:3478")]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """简化注释：处理连接状态变化，自动清理断开的会话"""
        logger.info("连接状态变为: %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if sessionid in nerfreals:
                logger.info(f"因连接状态为 {pc.connectionState}，正在关闭会话 {sessionid}")
                session_to_close = nerfreals.pop(sessionid, None)
                if session_to_close:
                    del session_to_close
                gc.collect()
            if pc in pcs:
                await pc.close()
                pcs.discard(pc)

    logger.info(f"会话 {sessionid}: 开始构建虚拟人实例...")
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    logger.info(f"会话 {sessionid}: 虚拟人实例构建完成。")

    if sessionid not in nerfreals:
        logger.warning(f"会话 {sessionid} 在实例准备好之前已关闭。")
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": "连接在设置过程中失败"})
        )
    nerfreals[sessionid] = nerfreal

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    # 简化注释：设置视频编解码器偏好，提高兼容性
    try:
        capabilities = RTCRtpSender.getCapabilities("video")
        if capabilities and capabilities.codecs:
            preferences = [codec for codec in capabilities.codecs if 'H264' in codec.mimeType]
            preferences += [codec for codec in capabilities.codecs if 'VP8' in codec.mimeType]
            video_transceiver = next((t for t in pc.getTransceivers() if t.sender and t.sender.track and t.sender.track.kind == "video"), None)
            if video_transceiver and preferences:
                video_transceiver.setCodecPreferences(preferences)
                logger.info("成功设置视频编解码器偏好为 H264 和 VP8。")
    except Exception as e:
        logger.warning(f"设置编解码器偏好时出错: {e}")

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})
    )

async def human(request):
    """简化注释：处理文本输入，驱动虚拟人进行交谈或复述"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if not sessionid or sessionid not in nerfreals:
            raise ValueError("无效或已过期的 sessionid")

        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        if params['type'] == 'echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type'] == 'chat':
            # --- 核心修复：流式处理LLM响应 ---
            async def stream_llm_to_tts():
                """简化注释：异步任务，将LLM的流式响应实时送入TTS"""
                try:
                    # 简化注释：使用异步for循环处理来自llm_client.ask的文本块
                    async for text_chunk in llm_client.ask(params['text']):
                        if sessionid in nerfreals:
                            # 简化注释：将收到的文本块送入虚拟人进行语音合成
                            nerfreals[sessionid].put_msg_txt(text_chunk)
                        else:
                            logger.warning(f"会话 {sessionid} 在接收LLM响应时已关闭，中断任务。")
                            break # 简化注释：如果会话关闭，则退出循环
                except Exception as e:
                    logger.error(f"stream_llm_to_tts 任务执行出错 (会话 {sessionid}): {e}")

            # 简化注释：创建后台任务处理流式响应，立即返回HTTP响应，不阻塞
            asyncio.create_task(stream_llm_to_tts())

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"})
        )
    except Exception as e:
        logger.exception('human接口异常:')
        return web.Response(
            content_type="application/json", status=500,
            text=json.dumps({"code": -1, "msg": str(e)})
        )

async def interrupt_talk(request):
    """简化注释：中断虚拟人当前的语音播报"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            nerfreals[sessionid].flush_talk()
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg":"ok"}))
    except Exception as e:
        logger.exception('interrupt_talk接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))

async def humanaudio(request):
    """简化注释：处理上传的音频文件，驱动虚拟人"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        if sessionid in nerfreals:
            fileobj = form["file"]
            filebytes = fileobj.file.read()
            nerfreals[sessionid].put_audio_file(filebytes)
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg":"ok"}))
    except Exception as e:
        logger.exception('humanaudio接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))

async def set_audiotype(request):
    """简化注释：设置音频类型或重新初始化虚拟人状态"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            nerfreals[sessionid].set_custom_state(params['audiotype'], params['reinit'])
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg":"ok"}))
    except Exception as e:
        logger.exception('set_audiotype接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))

async def record(request):
    """简化注释：控制录制功能的开始和结束"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        if sessionid in nerfreals:
            if params['type'] == 'start_record':
                nerfreals[sessionid].start_recording()
            elif params['type'] == 'end_record':
                nerfreals[sessionid].stop_recording()
        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg":"ok"}))
    except Exception as e:
        logger.exception('record接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))

async def is_speaking(request):
    """简化注释：查询虚拟人当前是否正在讲话"""
    params = await request.json()
    sessionid = int(params.get('sessionid', 0))
    speaking = nerfreals[sessionid].is_speaking() if sessionid in nerfreals else False
    return web.Response(content_type="application/json", text=json.dumps({"code": 0, "data": speaking}))

async def transcribe_audio(audio_bytes: bytes) -> str:
    """简化注释：将音频字节流通过FunASR转录为文本"""
    loop = asyncio.get_event_loop()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        generate_fn = partial(asr_model.generate, input=tmp_path, cache={}, language="zn+en", use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15)
        res = await loop.run_in_executor(None, generate_fn)

        # 简化注释：兼容新旧版FunASR的返回格式
        res = res[0] if isinstance(res, list) and res else res if isinstance(res, dict) else {}

        text_raw = res.get("text", "")
        if text_raw:
            processed_text = rich_transcription_postprocess(text_raw)
            logger.info(f"ASR 转写结果: {processed_text}")
            return processed_text
        return ""
    except Exception as e:
        logger.error(f"ASR转写出错: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

async def audio_chat(request):
    """简化注释：处理完整的语音聊天流程（ASR -> LLM -> TTS）"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        if not sessionid or sessionid not in nerfreals:
            return web.Response(content_type="application/json", status=400, text=json.dumps({"code": -1, "msg": "需要有效的sessionid"}))

        fileobj = form.get("file")
        if not fileobj:
            return web.Response(content_type="application/json", status=400, text=json.dumps({"code": -1, "msg": "需要音频文件"}))

        audio_bytes = fileobj.file.read()

        transcribed_text = await transcribe_audio(audio_bytes)
        if not transcribed_text:
            return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ASR未能识别出文本。"}))

        # --- 核心修复：流式处理LLM响应 ---
        async def stream_llm_to_tts():
            """简化注释：异步任务，将LLM的流式响应实时送入TTS"""
            try:
                async for text_chunk in llm_client.ask(transcribed_text):
                    if sessionid in nerfreals:
                        nerfreals[sessionid].put_msg_txt(text_chunk)
                    else:
                        logger.warning(f"会话 {sessionid} 在接收LLM响应时已关闭，中断任务。")
                        break
            except Exception as e:
                logger.error(f"audio_chat->stream_llm_to_tts 任务出错 (会话 {sessionid}): {e}")

        asyncio.create_task(stream_llm_to_tts())

        return web.Response(content_type="application/json", text=json.dumps({"code": 0, "msg": "ok"}))
    except Exception as e:
        logger.exception('audio_chat接口异常:')
        return web.Response(content_type="application/json", status=500, text=json.dumps({"code": -1, "msg": str(e)}))

async def on_shutdown(app):
    """简化注释：服务器关闭时，清理所有WebRTC连接"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    """简化注释：一个简单的异步POST请求函数"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=30) as response:
                return await response.text()
    except Exception as e:
        logger.info(f'POST请求错误: {e}')

async def run_push_session(push_url, sessionid):
    """简化注释：为rtcpush模式运行一个独立的虚拟人会话"""
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal
    pc = RTCPeerConnection()
    pcs.add(pc)
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    await pc.setLocalDescription(await pc.createOffer())
    answer_sdp = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type='answer'))

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    # --- 参数定义部分保持不变 ---
    parser.add_argument('--fps', type=int, default=50, help="视频帧率")
    parser.add_argument('-l', type=int, default=10, help="滑动窗口左侧长度 (单位: 20ms)")
    parser.add_argument('-m', type=int, default=8, help="滑动窗口中间长度 (单位: 20ms)")
    parser.add_argument('-r', type=int, default=10, help="滑动窗口右侧长度 (单位: 20ms)")
    parser.add_argument('--W', type=int, default=450, help="GUI 宽度")
    parser.add_argument('--H', type=int, default=450, help="GUI 高度")
    parser.add_argument('--batch_size', type=int, default=1, help="推理批次大小, MuseTalk建议为1")
    parser.add_argument('--customvideo_config', type=str, default='', help="自定义动作json配置文件")
    parser.add_argument('--tts', type=str, default='edgetts', help="TTS服务类型 (e.g., edgetts, xtts, gpt-sovits)")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural", help="TTS参考音频或说话人")
    parser.add_argument('--REF_TEXT', type=str, default=None, help="TTS参考文本")
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880', help="TTS服务地址")
    parser.add_argument('--model', type=str, default='musetalk', help="使用的模型 (musetalk, wav2lip, ultralight)")
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="定义 data/avatars 中的形象ID")
    parser.add_argument('--transport', type=str, default='webrtc', help="传输模式 (webrtc, rtcpush, virtualcam)")
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream', help="rtcpush模式下的推流地址")
    parser.add_argument('--max_session', type=int, default=1, help="最大会话数")
    parser.add_argument('--listenport', type=int, default=8010, help="Web服务监听端口")
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434/api/chat', help="Ollama聊天API的URL")
    parser.add_argument('--ollama-model', type=str, default='gemma3:1b', help="在Ollama中使用的模型名称")
    parser.add_argument('--ollama-system-prompt', type=str, default='你是AI数字人，请你简短回复，禁止输出表情符号。/nothink', help="给Ollama模型的系统提示")
    opt = parser.parse_args()

    # 简化注释：初始化 LLM 客户端
    llm_client = LLMClient(url=opt.ollama_url, model=opt.ollama_model, system_prompt=opt.ollama_system_prompt)

    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r', encoding='utf-8') as file:
            opt.customopt = json.load(file)
    else:
        opt.customopt = []

    # --- 模型加载部分保持不变 ---
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # 简化注释：初始化 FunASR 模型
    logger.info('正在加载 FunASR 模型...')
    try:
        asr_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('FunASR 模型加载成功。')
    except Exception as e:
        logger.error(f'加载 FunASR 模型失败: {e}')

    if opt.transport == 'virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    # --- aiohttp 应用设置和路由注册 ---
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/audio_chat", audio_chat) # 简化注释：注册语音聊天路由
    appasync.router.add_static('/', path='web')

    cors = aiohttp_cors.setup(appasync, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp': pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush': pagename = 'rtcpushapi.html'

    logger.info(f'HTTP服务器已启动; http://<serverip>:{opt.listenport}/{pagename}')
    logger.info(f'推荐访问WebRTC集成前端: http://<serverip>:{opt.listenport}/dashboard.html')

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())

        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url + (str(k) if k != 0 else "")
                loop.run_until_complete(run_push_session(push_url, k))

        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("正在关闭服务器...")

    runner = web.AppRunner(appasync)
    run_server(runner)