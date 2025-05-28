# video_processor.py (修正版)
import os
import cv2
import base64
import shutil
import asyncio
from moviepy import VideoFileClip
from openai import OpenAI, AsyncOpenAI
from datetime import datetime
from typing import List, Optional

# 导入新的配置项
from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE_URL,
    VIDEO_SUMMARY_MIN_DURATION_SEC,
    VIDEO_SUMMARY_MIN_TRANSCRIPT_WORDS,
    VIDEO_SUMMARY_NUM_IMAGES
)

# 初始化OpenAI客户端，传入自定义的 base_url
# 同步客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL
)
# 异步客户端
async_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL
)

def _extract_audio_sync(video_path: str, audio_output_path: str):
    """同步地从视频中提取音轨"""
    video_clip = VideoFileClip(video_path)
    audio_exists = video_clip.audio is not None
    if audio_exists:
        # 最简单的调用方式
        video_clip.audio.write_audiofile(audio_output_path)
    video_clip.close()
    return audio_output_path if audio_exists else None

async def extract_audio(video_path: str, temp_dir: str) -> Optional[str]:
    """异步包装器：从视频中提取音轨"""
    audio_output_path = os.path.join(temp_dir, f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3")
    return await asyncio.to_thread(_extract_audio_sync, video_path, audio_output_path)

def _transcribe_audio_sync(audio_path: str) -> str:
    """同步地将音频转录为文本"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

async def transcribe_audio(audio_path: str) -> str:
    """异步包装器：将音频转录为文本"""
    if not audio_path:
        return ""
    return await asyncio.to_thread(_transcribe_audio_sync, audio_path)

def _extract_frames_sync(video_path: str, num_frames: int, temp_dir: str) -> List[str]:
    """同步地从视频中提取均匀分布的帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_paths = []

    if frame_count < num_frames:
        num_frames = frame_count
        frame_indices = list(range(frame_count))
    else:
        interval = max(1, frame_count // (num_frames + 1))
        frame_indices = [i * interval for i in range(1, num_frames + 1)]

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            image_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
    
    cap.release()
    return image_paths

async def extract_frames(video_path: str, num_frames: int, temp_dir: str) -> List[str]:
    """异步包装器：从视频中提取均匀分布的帧"""
    return await asyncio.to_thread(_extract_frames_sync, video_path, num_frames, temp_dir)

def _encode_image_to_base64(image_path: str) -> str:
    """将图片编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def summarize_with_llm(transcription_text: str, image_paths: List[str]) -> str:
    """
    使用大型语言模型（LLM）总结视频内容
    """
    messages_content = []

    # 添加文字转录内容
    if transcription_text:
        messages_content.append({"type": "text", "text": f"视频文字转录内容：\n{transcription_text}\n\n"})
    
    # 添加图片内容
    if image_paths:
        messages_content.append({"type": "text", "text": "以下是视频关键帧图片，请结合图片内容进行总结："})
        for img_path in image_paths:
            base64_image = await asyncio.to_thread(_encode_image_to_base64, img_path)
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    # 添加总结指令
    messages_content.append({
        "type": "text",
        "text": "请根据以上提供的视频文字转录内容和关键帧图片，生成一个精简、流畅且具有吸引力的视频内容总结，字数控制在50-100字之间，用于评论区置顶展示。"
    })

    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个视频内容总结助手，能够结合文字和图片信息，生成简洁的视频摘要。"},
            {"role": "user", "content": messages_content}
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content

async def process_video_for_summary(video_id: int, video_db_path: str) -> Optional[str]:
    """
    主视频总结流程
    """
    temp_dir = os.path.join("temp_video_summary", str(video_id))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 获取视频时长
        def get_video_duration(path):
            video_clip = VideoFileClip(path)
            duration = video_clip.duration
            video_clip.close()
            return duration
        
        duration = await asyncio.to_thread(get_video_duration, video_db_path)

        # 检查视频时长是否满足条件
        if duration < VIDEO_SUMMARY_MIN_DURATION_SEC:
            print(f"Video {video_id} duration ({duration}s) is less than {VIDEO_SUMMARY_MIN_DURATION_SEC}s. Skipping summary.")
            return None

        # 1. 提取音轨
        audio_path = await extract_audio(video_db_path, temp_dir)

        # 2. 转录音轨
        transcription = ""
        if audio_path:
            transcription = await transcribe_audio(audio_path)
        
        # 检查转录字数是否满足条件
        word_count = len(transcription.split()) if transcription else 0
        if word_count < VIDEO_SUMMARY_MIN_TRANSCRIPT_WORDS:
            print(f"Video {video_id} transcript word count ({word_count}) is less than {VIDEO_SUMMARY_MIN_TRANSCRIPT_WORDS}. Skipping summary.")
            return None

        # 3. 截取图片
        image_paths = await extract_frames(video_db_path, VIDEO_SUMMARY_NUM_IMAGES, temp_dir)

        # 4. 调用大模型总结
        summary = await summarize_with_llm(transcription, image_paths)
        return summary

    except Exception as e:
        print(f"Error processing video {video_id} for summary: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)