# app.py
# 导入必要的库
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
from functools import partial # 新增：导入 partial 以便包装函数调用
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

# 新增：添加 FunASR 和其他必要的导入
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile
import os

# 新增：从 llm.py 导入 LLMClient
from llm import LLMClient


app = Flask(__name__)
nerfreals:Dict[int, BaseReal] = {} # sessionid:BaseReal
opt = None
model = None
avatar = None
asr_model = None # 新增：ASR 模型全局变量
llm_client = None # 新增：LLM 客户端全局变量


##### webrtc ###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    """根据命令行参数构建并返回一个虚拟人实例"""
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
        pass
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

async def offer(request):
    """处理 WebRTC offer 请求，建立点对点连接"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = randN(6)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    
    # --- 合并修改：遵从用户要求，使用指定的 STUN 服务器 ---
    ice_servers = [
        RTCIceServer(urls="stun:stun.miwifi.com:3478"),
    ]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """连接状态改变时的回调函数"""
        logger.info("Connection state is %s" % pc.connectionState)
        # --- 关键修改：保留更健壮的状态处理和资源清理逻辑 ---
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if sessionid in nerfreals:
                logger.info(f"Closing session {sessionid} due to connection state: {pc.connectionState}")
                # 安全地删除会话实例
                session_to_close = nerfreals.pop(sessionid, None)
                if session_to_close:
                    del session_to_close
                gc.collect() # 执行垃圾回收
            if pc in pcs:
                await pc.close()
                pcs.discard(pc)

    # 异步构建虚拟人实例，避免阻塞
    logger.info(f"Session {sessionid}: Starting to build NeRFReal instance...")
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    logger.info(f"Session {sessionid}: NeRFReal instance built.")
    
    # 检查会话是否因连接失败而提前关闭
    if sessionid not in nerfreals:
        logger.warning(f"Session {sessionid} was closed before NeRFReal instance was ready.")
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps({"code": -1, "msg": "Connection failed during setup"}),
        )
    nerfreals[sessionid] = nerfreal

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    
    # --- 关键修复：放宽视频编解码器偏好以提高兼容性 ---
    try:
        capabilities = RTCRtpSender.getCapabilities("video")
        if capabilities and capabilities.codecs:
            # 优先选择 H264，但同时保留 VP8 作为备选方案，增加连接成功率
            preferences = [codec for codec in capabilities.codecs if 'H264' in codec.mimeType]
            preferences += [codec for codec in capabilities.codecs if 'VP8' in codec.mimeType]
            
            video_transceiver = next((t for t in pc.getTransceivers() if t.sender and t.sender.track and t.sender.track.kind == "video"), None)
            
            if video_transceiver and preferences:
                video_transceiver.setCodecPreferences(preferences)
                logger.info("Successfully set video codec preferences to H264 and VP8.")
            else:
                logger.warning("Could not find video transceiver or suitable codecs to set preferences.")
    except Exception as e:
        logger.warning(f"An error occurred while setting codec preferences: {e}")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    """处理文本输入，用于驱动虚拟人（echo 或 chat 模式）"""
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid', 0))
        if not sessionid or sessionid not in nerfreals:
            raise ValueError("Invalid or expired sessionid")

        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type']=='chat':
            async def chat_task():
                # --- 核心改动：调用 llm_client.ask ---
                llm_response_text = await llm_client.ask(params['text'])
                if sessionid in nerfreals:
                    nerfreals[sessionid].put_msg_txt(llm_response_text)
                else:
                    logger.warning(f"会话 {sessionid} 在 LLM 响应后已关闭。")
            
            asyncio.create_task(chat_task())

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def interrupt_talk(request):
    """中断当前虚拟人的播报"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid',0))
        if sessionid in nerfreals:
            nerfreals[sessionid].flush_talk()
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def humanaudio(request):
    """处理音频文件上传，用于驱动虚拟人"""
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        if sessionid in nerfreals:
            fileobj = form["file"]
            filebytes=fileobj.file.read()
            nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    """设置音频类型或重新初始化"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid',0))
        if sessionid in nerfreals:
            nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    """控制录制开始和结束"""
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid',0))
        if sessionid in nerfreals:
            if params['type']=='start_record':
                nerfreals[sessionid].start_recording()
            elif params['type']=='end_record':
                nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    """查询虚拟人当前是否正在讲话"""
    params = await request.json()
    sessionid = int(params.get('sessionid',0))
    speaking = False
    if sessionid in nerfreals:
        speaking = nerfreals[sessionid].is_speaking()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": speaking}
        ),
    )

# --- 核心修复：替换为兼容新旧版 FunASR 的完整函数 ---
async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    将音频字节流转录为文本。
    这是一个 I/O 和 CPU 密集型操作的混合体，因此使用临时文件和执行器。
    它现在可以处理新版(>=1.0.3)FunASR返回的列表和旧版返回的字典。
    """
    loop = asyncio.get_event_loop()
    tmp_path = None
    try:
        # 创建一个带 .wav 后缀的临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 使用 functools.partial 将调用包装为关键字参数形式
        generate_fn = partial(
            asr_model.generate,
            input=tmp_path,
            cache={},
            language="zn+en",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15
        )

        # 在后台线程中执行 ASR 函数
        res = await loop.run_in_executor(None, generate_fn)

        # 关键修复：兼容新版 list / 旧版 dict
        # 如果返回的是列表，则取第一个元素；如果列表为空，则返回空字典
        if isinstance(res, list):
            res = res[0] if res else {}

        # 安全地获取文本并进行后处理
        text_raw = res.get("text", "")
        if text_raw:
            processed_text = rich_transcription_postprocess(text_raw)
            logger.info(f"ASR 转写结果: {processed_text}")
            return processed_text
        else:
            logger.warning("ASR 转写未返回任何文本。")
            return ""

    except Exception as e:
        logger.error(f"ASR 转写过程中发生错误: {e}")
        return ""
    finally:
        # 确保清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 核心改动：删除旧的 query_ollama 函数 ---

# 新增：定义新的语音聊天端点
async def audio_chat(request):
    """处理语音聊天请求：ASR -> LLM -> TTS"""
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        
        if not sessionid or sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                status=400,
                text=json.dumps({"code": -1, "msg": "valid sessionid is required"}),
            )
        
        fileobj = form.get("file")
        if not fileobj:
            return web.Response(
                content_type="application/json",
                status=400,
                text=json.dumps({"code": -1, "msg": "audio file is required"}),
            )
            
        audio_bytes = fileobj.file.read()
        
        # 1. 音频转文本 (ASR)
        transcribed_text = await transcribe_audio(audio_bytes)
        if not transcribed_text:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "msg": "ASR failed or produced no text."}),
            )

        # 2. 文本送入 LLM
        # --- 核心改动：调用 llm_client.ask ---
        llm_response_text = await llm_client.ask(transcribed_text)

        # 3. LLM 响应送入虚拟人 TTS
        nerfreal_session = nerfreals.get(sessionid)
        if nerfreal_session:
            nerfreal_session.put_msg_txt(llm_response_text)
        else:
            logger.error(f"Session {sessionid} not found in nerfreals after processing.")

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        logger.exception('Exception in /audio_chat:')
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps({"code": -1, "msg": str(e)}),
        )

async def on_shutdown(app):
    """服务器关闭时，清理所有 WebRTC 连接"""
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    """一个简单的异步 POST 请求函数"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data, timeout=30) as response:
                return await response.text()
    except Exception as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    """为 rtcpush 模式运行一个独立的虚拟人会话"""
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################

if __name__ == '__main__':
    # 设置多进程启动方法为 'spawn'，以兼容不同平台
    # 在Windows或macOS上，使用多进程时这通常是必须的
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser()
    
    # --- 通用参数 ---
    parser.add_argument('--fps', type=int, default=25, help="视频帧率")
    parser.add_argument('-l', type=int, default=10, help="滑动窗口左侧长度 (单位: 20ms)")
    parser.add_argument('-m', type=int, default=8, help="滑动窗口中间长度 (单位: 20ms)")
    parser.add_argument('-r', type=int, default=10, help="滑动窗口右侧长度 (单位: 20ms)")
    parser.add_argument('--W', type=int, default=450, help="GUI 宽度")
    parser.add_argument('--H', type=int, default=450, help="GUI 高度")
    parser.add_argument('--batch_size', type=int, default=1, help="推理批次大小, MuseTalk建议为1")
    parser.add_argument('--customvideo_config', type=str, default='', help="自定义动作json配置文件")
    
    # --- TTS 相关参数 ---
    parser.add_argument('--tts', type=str, default='edgetts', help="TTS服务类型 (e.g., edgetts, xtts, gpt-sovits)")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural", help="TTS参考音频或说话人")
    parser.add_argument('--REF_TEXT', type=str, default=None, help="TTS参考文本")
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880', help="TTS服务地址")

    # --- 模型与传输参数 ---
    parser.add_argument('--model', type=str, default='musetalk', help="使用的模型 (musetalk, wav2lip, ultralight)")
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="定义 data/avatars 中的形象ID")
    parser.add_argument('--transport', type=str, default='webrtc', help="传输模式 (webrtc, rtcpush, virtualcam)")
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream', help="rtcpush模式下的推流地址")
    
    # --- 服务器参数 ---
    parser.add_argument('--max_session', type=int, default=1, help="最大会话数")
    parser.add_argument('--listenport', type=int, default=8010, help="Web服务监听端口")

    # 新增：Ollama 配置参数
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434/api/chat', help="Ollama聊天API的URL")
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b', help="在Ollama中使用的模型名称")
    parser.add_argument('--ollama-system-prompt', type=str, default='你是AI数字人，请你简短回复，禁止输出表情符号。/nothink', help="给Ollama模型的系统提示")
    
    opt = parser.parse_args()
    
    # --- 核心改动：初始化 LLMClient ---
    llm_client = LLMClient(
        url=opt.ollama_url,
        model=opt.ollama_model,
        system_prompt=opt.ollama_system_prompt
    )

    # 加载自定义动作配置
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r', encoding='utf-8') as file:
            opt.customopt = json.load(file)

    # 根据选择加载不同的虚拟人模型
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)
    
    # 新增：初始化 FunASR 模型
    logger.info('Loading FunASR model...')
    try:
        model_dir = "iic/SenseVoiceSmall"
        asr_model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        logger.info('FunASR model loaded successfully.')
    except Exception as e:
        logger.error(f'Failed to load FunASR model: {e}')
        # exit(1) # 如果需要，可以在模型加载失败时退出

    # 为 virtualcam 模式启动渲染线程
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # 设置 aiohttp 应用
    appasync = web.Application(client_max_size=1024**2*100) # 允许最大100MB的请求体
    appasync.on_shutdown.append(on_shutdown)
    
    # 注册所有API路由
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/audio_chat", audio_chat) # 新增：注册语音聊天路由
    appasync.router.add_static('/',path='web') # 提供静态文件服务

    # 配置CORS，允许跨域请求
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
        })
    for route in list(appasync.router.routes()):
        cors.add(route)

    # 根据传输模式决定默认页面
    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    
    logger.info('Start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    
    # 启动服务器的函数
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        
        # 如果是 rtcpush 模式，提前创建会话
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
                
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # 清理工作
            logger.info("Shutting down server...")
            # 可以添加其他清理任务
            
    runner = web.AppRunner(appasync)
    run_server(runner)