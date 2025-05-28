# background_tasks.py (修正版)
import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List
import os

from database import SessionLocal
import models
from video_processor import process_video_for_summary
# 导入配置
from config import SUMMARY_CHECK_INTERVAL_MINUTES, SUMMARY_MAX_VIDEOS_PER_BATCH

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSummaryScheduler:
    def __init__(self, check_interval_minutes: int = None):
        """
        视频摘要调度器
        
        Args:
            check_interval_minutes: 检查间隔（分钟），如果为None则使用配置文件中的值
        """
        # 使用配置文件中的值
        self.check_interval = (check_interval_minutes or SUMMARY_CHECK_INTERVAL_MINUTES) * 60  # 转换为秒
        self.is_running = False
        
    async def get_videos_without_summary(self, db: Session) -> List[models.Video]:
        """
        获取没有摘要的视频列表
        """
        # 查询所有没有摘要的视频
        videos_without_summary = db.query(models.Video).outerjoin(
            models.VideoSummary, 
            models.Video.vid == models.VideoSummary.vid
        ).filter(
            models.VideoSummary.vid.is_(None)  # 没有对应的摘要记录
        ).all()
        
        logger.info(f"Found {len(videos_without_summary)} videos without summary")
        return videos_without_summary
    
    async def process_single_video(self, video: models.Video) -> bool:
        """
        处理单个视频的摘要生成
        为每个视频创建独立的数据库会话
        
        Returns:
            bool: 是否成功生成摘要
        """
        # 为每个视频创建独立的数据库会话
        db = SessionLocal()
        try:
            # 构建视频文件的完整路径
            video_full_path = os.path.join(os.getcwd(), "static", "videos", video.video_url)
            
            if not os.path.exists(video_full_path):
                logger.warning(f"Video file not found: {video_full_path}")
                return False
            
            logger.info(f"Processing video {video.vid}: {video.title}")
            
            # 再次检查该视频是否已经有摘要（防止并发问题）
            existing_summary = db.query(models.VideoSummary).filter(models.VideoSummary.vid == video.vid).first()
            if existing_summary:
                logger.info(f"Video {video.vid} already has summary, skipping")
                return False
            
            # 调用视频处理函数
            summary = await process_video_for_summary(video.vid, video_full_path)
            
            if summary:
                # 保存摘要到数据库
                new_summary = models.VideoSummary(vid=video.vid, summary=summary)
                db.add(new_summary)
                db.commit()
                db.refresh(new_summary)
                
                logger.info(f"Successfully generated and saved summary for video {video.vid}: {summary[:100]}...")
                return True
            else:
                logger.info(f"Video {video.vid} did not meet summary criteria")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video {video.vid}: {e}")
            db.rollback()  # 回滚事务
            import traceback
            traceback.print_exc()
            return False
        finally:
            db.close()  # 确保数据库会话被关闭
    
    async def run_summary_batch(self, max_videos_per_batch: int = None):
        """
        运行一批视频摘要任务
        
        Args:
            max_videos_per_batch: 每批处理的最大视频数量，如果为None则使用配置文件中的值
        """
        # 使用配置文件中的值
        max_videos = max_videos_per_batch or SUMMARY_MAX_VIDEOS_PER_BATCH
        
        # 创建一个会话来查询需要处理的视频
        db = SessionLocal()
        try:
            # 获取需要处理的视频
            videos_to_process = await self.get_videos_without_summary(db)
            
            if not videos_to_process:
                logger.info("No videos need summary processing")
                return
            
            # 限制每批处理的数量
            videos_batch = videos_to_process[:max_videos]
            logger.info(f"Processing batch of {len(videos_batch)} videos (max: {max_videos})")
            
        except Exception as e:
            logger.error(f"Error getting videos to process: {e}")
            return
        finally:
            db.close()  # 关闭查询用的会话
        
        # 处理每个视频（每个视频使用独立的会话）
        success_count = 0
        for i, video in enumerate(videos_batch):
            try:
                logger.info(f"Processing video {i+1}/{len(videos_batch)}: {video.vid}")
                success = await self.process_single_video(video)
                if success:
                    success_count += 1
                
                # 在视频之间添加短暂延迟，避免API调用过于频繁
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in batch processing for video {video.vid}: {e}")
                continue
        
        logger.info(f"Batch completed: {success_count}/{len(videos_batch)} videos processed successfully")
    
    async def start_scheduler(self, max_videos_per_batch: int = None):
        """
        启动调度器
        """
        self.is_running = True
        logger.info(f"Video summary scheduler started. Check interval: {self.check_interval/60} minutes, Max videos per batch: {max_videos_per_batch or SUMMARY_MAX_VIDEOS_PER_BATCH}")
        
        while self.is_running:
            try:
                logger.info("Starting scheduled video summary check...")
                await self.run_summary_batch(max_videos_per_batch)
                logger.info(f"Scheduled check completed. Next check in {self.check_interval/60} minutes")
                
                # 等待下一次检查
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                import traceback
                traceback.print_exc()
                # 即使出错也要继续运行，等待一段时间后重试
                await asyncio.sleep(60)  # 出错时等待1分钟后重试
    
    def stop_scheduler(self):
        """
        停止调度器
        """
        self.is_running = False
        logger.info("Video summary scheduler stopped")

# 全局调度器实例 - 使用配置文件中的默认值
video_summary_scheduler = VideoSummaryScheduler()