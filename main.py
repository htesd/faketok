import re

from aiohttp.web_fileresponse import FileResponse
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Depends, Header,Request
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import os
import shutil
import cv2
import numpy as np
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

# 数据库模型和连接配置
from database import SessionLocal, engine
import models, schemas
from contextlib import asynccontextmanager
from background_tasks import video_summary_scheduler
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("Starting FastAPI application...")
    
    # 确保所有目录存在
    os.makedirs("static/videos", exist_ok=True)
    os.makedirs("static/covers", exist_ok=True)
    os.makedirs("static/avatars", exist_ok=True)
    os.makedirs("static/videos/resolutions", exist_ok=True)
    os.makedirs("temp_video_summary", exist_ok=True)
    
    # 创建数据库表
    models.Base.metadata.create_all(bind=engine)
    
    # 启动后台任务调度器 - 使用配置文件中的默认值
    asyncio.create_task(video_summary_scheduler.start_scheduler())

    
    yield
    

    video_summary_scheduler.stop_scheduler()
    

app = FastAPI(title="短视频应用API")
# 添加CORS中间件（如果需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/test")
async def test():
    return "hello world!"


@app.post("/user/login")
async def user_auth(
        username: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    # 先尝试登录
    existing_user = db.query(models.User).filter(models.User.username == username).first()

    if existing_user:
        # 用户存在，验证密码
        if existing_user.password == password:
            return {
                "success": True,
                "uid": existing_user.uid,
                "message": 1  # 登录成功
            }
        else:
            return {
                "success": False,
                "uid": None,
                "message": 2  # 账号密码不匹配
            }
    else:
        # 用户不存在，自动注册
        new_user = models.User(
            username=username,
            password=password,
            nickname=username,  # 默认昵称为用户名
            avatar_url="default_avatar.png"  # 默认头像
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "success": True,
            "uid": new_user.uid,
            "message": 0  # 注册成功
        }



@app.post("/user/register")
async def register_user(
        username: str = Form(...),
        password: str = Form(...),
        nickname: str = Form(None),
        db: Session = Depends(get_db)
):
    # 检查用户名是否已存在
    existing_user = db.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # 如果未提供昵称，使用用户名作为昵称
    if not nickname:
        nickname = username

    # 创建新用户
    new_user = models.User(
        username=username,
        password=password,  # 实际应用中应该哈希处理
        nickname=nickname
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "success": True,
        "uid": new_user.uid,
        "message": "注册成功"
    }

# 用户相关API
@app.get("/user/{uid}/info")
async def get_user_info(uid: int, request: Request, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 获取用户的作品、收藏、喜欢等信息
    videos = db.query(models.Video).filter(models.Video.uid == uid).all()
    favorites = db.query(models.Favorite).filter(models.Favorite.uid == uid).all()
    likes = db.query(models.Like).filter(models.Like.uid == uid).all()

    # 获取关注和粉丝
    following = db.query(models.Follow).filter(models.Follow.follower_id == uid).all()
    followers = db.query(models.Follow).filter(models.Follow.followed_id == uid).all()

    return {
        "uid": user.uid,
        "nickname": user.nickname,
        "bio": user.bio,
        "avatar_url": get_avatar_url(request, user.avatar_url),
        "videos": [v.vid for v in videos],
        "favorites": [f.vid for f in favorites],
        "likes": [l.vid for l in likes],
        "following": [f.followed_id for f in following],
        "followers": [f.follower_id for f in followers]
    }


# @app.post("/user/{uid}/profile")
# async def update_profile_image(
#         uid: int,
#         image: UploadFile = File(...),
#         request: Request = None,
#         db: Session = Depends(get_db)
# ):
#     """原有的一步式头像更新接口，保持向后兼容"""
#     user = db.query(models.User).filter(models.User.uid == uid).first()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
#
#     # 确保目录存在
#     os.makedirs("static/avatars", exist_ok=True)
#
#     # 保存上传的头像
#     file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
#     avatar_filename = f"{uid}_avatar.{file_extension}"
#     file_location = f"static/avatars/{avatar_filename}"
#
#     # 如果用户已有头像，删除旧文件
#     if user.avatar_url and user.avatar_url != "default_avatar.png":
#         old_file_path = f"static/avatars/{user.avatar_url}"
#         if os.path.exists(old_file_path):
#             os.remove(old_file_path)
#
#     # 保存原始文件
#     with open(file_location, "wb") as file_object:
#         shutil.copyfileobj(image.file, file_object)
#
#     # 使用OpenCV调整头像尺寸到300x300
#     img = cv2.imread(file_location)
#     if img is not None:
#         resized_img = cv2.resize(img, (300, 300))
#         cv2.imwrite(file_location, resized_img)
#
#     # 更新数据库中的头像文件名
#     user.avatar_url = avatar_filename
#     db.commit()
#
#     return {"success": True, "avatar_url": avatar_filename}

@app.post("/upload/avatar")
async def upload_avatar_file(
        image: UploadFile = File(...),
        request: Request = None
):
    # 确保目录存在
    os.makedirs("static/avatars", exist_ok=True)

    # 生成唯一的文件名（使用时间戳避免冲突）
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
    avatar_filename = f"temp_{timestamp}.{file_extension}"
    file_location = f"static/avatars/{avatar_filename}"

    # 保存原始文件
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(image.file, file_object)

    # 使用OpenCV调整头像尺寸到300x300
    img = cv2.imread(file_location)
    if img is not None:
        resized_img = cv2.resize(img, (300, 300))
        cv2.imwrite(file_location, resized_img)

    # 返回完整的URL
    base_url = str(request.base_url).rstrip('/')
    avatar_url = f"{base_url}/static/avatars/{avatar_filename}"

    return {
        "success": True,
        "avatar_url": avatar_url,
        "filename": avatar_filename,
        "message": "Avatar uploaded successfully"
    }


@app.post("/user/{uid}/confirm_avatar")
async def confirm_avatar(
        uid: int,
        avatar_url: str = Form(...),
        db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 从完整URL中提取文件名
    # 例如：http://106.13.81.108:8000/static/avatars/temp_20250518140000.jpg
    # 提取：temp_20250518140000.jpg
    if "/static/avatars/" in avatar_url:
        filename = avatar_url.split("/static/avatars/")[-1]
    else:
        raise HTTPException(status_code=400, detail="Invalid avatar URL")

    # 检查文件是否存在
    file_path = f"static/avatars/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Avatar file not found")

    # 如果是临时文件，重命名为用户专用文件名
    if filename.startswith("temp_"):
        file_extension = filename.split('.')[-1]
        new_filename = f"{uid}_avatar.{file_extension}"
        new_file_path = f"static/avatars/{new_filename}"

        # 如果用户已有头像，删除旧文件
        if user.avatar_url and user.avatar_url != "default_avatar.png":
            old_file_path = f"static/avatars/{user.avatar_url}"
            if os.path.exists(old_file_path):
                os.remove(old_file_path)

        # 重命名文件
        os.rename(file_path, new_file_path)
        final_filename = new_filename
    else:
        final_filename = filename

    # 更新数据库中的头像文件名
    user.avatar_url = final_filename
    db.commit()

    return {
        "success": True,
        "avatar_url": final_filename,
        "message": "Avatar updated successfully"
    }


@app.delete("/upload/avatar/cleanup")
async def cleanup_temp_avatars():
    """清理超过1小时的临时头像文件"""
    avatar_dir = "static/avatars"
    if not os.path.exists(avatar_dir):
        return {"success": True, "message": "No temp files to clean"}

    current_time = datetime.now()
    cleaned_count = 0

    for filename in os.listdir(avatar_dir):
        if filename.startswith("temp_"):
            file_path = os.path.join(avatar_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))

            # 如果文件超过1小时，删除它
            if (current_time - file_time).total_seconds() > 3600:
                os.remove(file_path)
                cleaned_count += 1

    return {
        "success": True,
        "message": f"Cleaned {cleaned_count} temporary avatar files"
    }

from fastapi.responses import Response
import mimetypes


# 辅助函数：处理头像URL
def get_avatar_url(request: Request, avatar_filename: str) -> str:
    if avatar_filename == "default_avatar.png":
        # 返回默认头像的完整URL
        base_url = str(request.base_url).rstrip('/')
        return f"{base_url}/static/avatars/default_avatar.png"
    else:
        # 返回用户头像的完整URL
        base_url = str(request.base_url).rstrip('/')
        return f"{base_url}/static/avatars/{avatar_filename}"


# 修复辅助函数：处理文件URL
def get_file_url(request: Request, file_path: str, file_type: str) -> str:
    base_url = str(request.base_url).rstrip('/')

    # 如果文件路径已经包含static/，直接使用
    if file_path.startswith('static/'):
        return f"{base_url}/{file_path}"

    # 否则根据文件类型添加相应的路径
    if file_type == "avatar":
        return f"{base_url}/static/avatars/{file_path}"
    elif file_type == "cover":
        return f"{base_url}/static/covers/{file_path}"
    elif file_type == "video":
        return f"{base_url}/static/videos/{file_path}"
    else:
        return f"{base_url}/static/{file_path}"


# 或者更简单的解决方案，专门处理视频和封面URL
def get_video_url(request: Request, video_filename: str) -> str:
    base_url = str(request.base_url).rstrip('/')
    # 移除可能存在的路径前缀
    if video_filename.startswith('static/videos/'):
        video_filename = video_filename[14:]  # 移除 'static/videos/'
    return f"{base_url}/static/videos/{video_filename}"


def get_cover_url(request: Request, cover_filename: str) -> str:
    base_url = str(request.base_url).rstrip('/')
    # 移除可能存在的路径前缀
    if cover_filename.startswith('static/covers/'):
        cover_filename = cover_filename[14:]  # 移除 'static/covers/'
    return f"{base_url}/static/covers/{cover_filename}"


def get_resolution_url(request: Request, resolution_filename: str) -> str:
    base_url = str(request.base_url).rstrip('/')
    # 移除可能存在的路径前缀
    if resolution_filename.startswith('static/videos/resolutions/'):
        resolution_filename = resolution_filename[26:]  # 移除 'static/videos/resolutions/'
    return f"{base_url}/static/videos/resolutions/{resolution_filename}"


# 头像获取接口（处理默认头像）
@app.get("/user/{uid}/avatar")
async def get_user_avatar(uid: int, request: Request, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        # 用户不存在，返回默认头像URL
        return {"avatar_url": get_avatar_url(request, "default_avatar.png")}

    return {"avatar_url": get_avatar_url(request, user.avatar_url)}

@app.post("/user/{uid}/nickname")
async def update_nickname(uid: int, nickname: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.nickname = nickname
    db.commit()
    return {"success": True}


@app.post("/user/{uid}/bio")
async def update_bio(uid: int, bio: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.bio = bio
    db.commit()
    return {"success": True}


@app.post("/user/{uid}/profile")
async def update_user_profile(
        uid: int,
        nickname: Optional[str] = Form(None),
        bio: Optional[str] = Form(None),
        avatar_url: Optional[str] = Form(None),  # 这里传入已经上传好的头像URL或文件名
        db: Session = Depends(get_db)
):
    """
    通用个人信息修改接口
    可以一次性修改昵称、简介、头像等信息
    头像需要先通过 /upload/avatar 接口上传，然后传入URL或文件名
    """
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updated_fields = []

    # 更新昵称
    if nickname is not None:
        user.nickname = nickname
        updated_fields.append("nickname")

    # 更新简介
    if bio is not None:
        user.bio = bio
        updated_fields.append("bio")

    # 更新头像
    if avatar_url is not None:
        # 处理头像URL，提取文件名
        if avatar_url.startswith("http"):
            # 如果传入的是完整URL，提取文件名
            if "/static/avatars/" in avatar_url:
                filename = avatar_url.split("/static/avatars/")[-1]
            else:
                raise HTTPException(status_code=400, detail="Invalid avatar URL format")
        else:
            # 如果传入的就是文件名
            filename = avatar_url

        # 检查文件是否存在
        file_path = f"static/avatars/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Avatar file not found")

        # 如果是临时文件，重命名为用户专用文件名
        if filename.startswith("temp_"):
            file_extension = filename.split('.')[-1]
            new_filename = f"{uid}_avatar.{file_extension}"
            new_file_path = f"static/avatars/{new_filename}"

            # 如果用户已有头像，删除旧文件
            if user.avatar_url and user.avatar_url != "default_avatar.png":
                old_file_path = f"static/avatars/{user.avatar_url}"
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)

            # 重命名文件
            os.rename(file_path, new_file_path)
            user.avatar_url = new_filename
        else:
            # 如果不是临时文件，直接使用
            # 但需要验证文件名格式是否合理
            if filename != "default_avatar.png" and not filename.startswith(f"{uid}_"):
                # 为了安全，重命名文件
                file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
                new_filename = f"{uid}_avatar.{file_extension}"
                new_file_path = f"static/avatars/{new_filename}"

                # 删除旧头像
                if user.avatar_url and user.avatar_url != "default_avatar.png":
                    old_file_path = f"static/avatars/{user.avatar_url}"
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)

                # 复制文件到新位置
                shutil.copy2(file_path, new_file_path)
                user.avatar_url = new_filename
            else:
                user.avatar_url = filename

        updated_fields.append("avatar")

    # 提交更改到数据库
    db.commit()

    return {
        "success": True,
        "updated_fields": updated_fields,
        "user_info": {
            "uid": user.uid,
            "nickname": user.nickname,
            "bio": user.bio,
            "avatar_url": user.avatar_url
        },
        "message": f"Updated {', '.join(updated_fields) if updated_fields else 'no fields'}"
    }



# 视频相关API
@app.get("/video/{vid}/previewInfo", response_model=schemas.VideoPreviewInfo)
async def get_video_preview(vid: int,request: Request, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "vid": video.vid,
        "cover_url": get_file_url(request,video.cover_url,"cover"),
        "likes_count": video.likes_count
    }


@app.get("/video/{vid}/cover")
async def get_video_cover(vid: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    cover_path = video.cover_url

    # 如果是相对路径，转换为绝对路径
    if not cover_path.startswith(('http://', 'https://')):
        cover_path = os.path.join(os.getcwd(), cover_path)

    # 检查文件是否存在
    if not os.path.exists(cover_path):
        # 返回默认封面
        cover_path = os.path.join(os.getcwd(), "static/covers/default_cover.jpg")

    # 确定MIME类型
    content_type, _ = mimetypes.guess_type(cover_path)
    if not content_type:
        content_type = "image/jpeg"  # 默认为JPEG

    # 读取文件内容
    with open(cover_path, "rb") as f:
        file_content = f.read()

    # 返回响应
    return Response(content=file_content, media_type=content_type)


@app.get("/video/{vid}/stream")
async def stream_video(
        vid: int,
        resolution: Optional[str] = None,
        range: Optional[str] = Header(None),
        db: Session = Depends(get_db)
):
    # 获取视频信息
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 如果指定了分辨率，尝试获取对应分辨率的视频
    video_path = video.video_url
    if resolution:
        video_res = db.query(models.VideoResolution).filter(
            models.VideoResolution.vid == vid,
            models.VideoResolution.resolution == resolution
        ).first()

        if video_res:
            video_path = video_res.video_url

    # 如果是相对路径，转换为绝对路径
    if not video_path.startswith(('http://', 'https://')):
        video_path = os.path.join(os.getcwd(), video_path)

    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # 获取文件信息
    file_size = os.path.getsize(video_path)

    # 获取文件的MIME类型
    import mimetypes
    content_type = mimetypes.guess_type(video_path)[0] or 'video/mp4'

    # 处理范围请求 (用于视频流)
    start = 0
    end = file_size - 1
    status_code = 200

    # 更健壮的Range头解析
    if range:
        try:
            # 标准格式: "bytes=start-end"
            range_str = range.replace(' ', '')
            if range_str.startswith('bytes='):
                range_parts = range_str[6:].split('-')

                if range_parts[0]:
                    start = int(range_parts[0])

                if len(range_parts) > 1 and range_parts[1]:
                    end = min(int(range_parts[1]), file_size - 1)

                if start <= end:
                    status_code = 206  # Partial Content
                else:
                    # 范围无效，返回完整文件
                    start = 0
                    status_code = 200
        except ValueError:
            # 解析失败，返回完整文件
            start = 0
            status_code = 200

    # 计算内容长度
    content_length = end - start + 1

    # 设置响应头
    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(content_length),
        'Content-Type': content_type,
    }

    if status_code == 206:
        headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'

    # 创建文件响应，使用块读取以提高性能
    chunk_size = 1024 * 1024  # 1MB 块大小，可以根据需求调整

    async def send_chunks():
        with open(video_path, 'rb') as video_file:
            # 移动到正确的位置
            video_file.seek(start)

            bytes_sent = 0
            while bytes_sent < content_length:
                # 读取适当大小的块
                chunk_bytes = min(chunk_size, content_length - bytes_sent)
                data = video_file.read(chunk_bytes)

                if not data:
                    break

                bytes_sent += len(data)
                yield data

    return StreamingResponse(
        send_chunks(),
        status_code=status_code,
        headers=headers
    )

@app.get("/video/{vid}/info", response_model=schemas.VideoInfo)
async def get_video_info(vid: int, request: Request,db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 获取作者信息
    author = db.query(models.User).filter(models.User.uid == video.uid).first()

    return {
        "vid": video.vid,
        "author_uid": author.uid,
        "author_nickname": author.nickname,
        "author_avatar_url": get_file_url(request,author.avatar_url,"avatar"),
        "title": video.title,
        "description": video.description,
        "video_url": get_file_url(request,video.video_url,"video"),
        "cover_url": get_file_url(request,video.cover_url,"cover"),
        "likes_count": video.likes_count,
        "comments_count": video.comments_count,
        "favorites_count": video.favorites_count,
        "created_at": video.created_at
    }


@app.post("/video/upload")
async def upload_video(
        uid: int = Form(...),
        title: str = Form(...),
        description: str = Form(...),
        tags: str = Form(...),
        video: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    # 检查用户是否存在
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 确保目录存在
    os.makedirs("static/videos", exist_ok=True)
    os.makedirs("static/covers", exist_ok=True)

    # 保存视频文件
    video_filename = f"{uid}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{video.filename}"
    video_path = f"static/videos/{video_filename}"
    with open(video_path, "wb") as file_object:
        shutil.copyfileobj(video.file, file_object)

    # 使用OpenCV生成封面并缩放到0.5
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cover_filename = f"{video_filename.split('.')[0]}_cover.jpg"
    cover_path = f"static/covers/{cover_filename}"

    if ret:
        # 获取原始尺寸并缩放到0.5
        height, width = frame.shape[:2]
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imwrite(cover_path, resized_frame)
    cap.release()

    # 创建视频记录（只存储文件名，不包含路径）
    new_video = models.Video(
        uid=uid,
        title=title,
        description=description,
        video_url=video_filename,  # 只存储文件名
        cover_url=cover_filename  # 只存储文件名
    )
    db.add(new_video)
    db.commit()
    db.refresh(new_video)

    # 处理标签
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    for tag_name in tag_list:
        tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
        if not tag:
            tag = models.Tag(name=tag_name)
            db.add(tag)
            db.commit()
            db.refresh(tag)

        video_tag = models.VideoTag(vid=new_video.vid, tag_id=tag.tag_id)
        db.add(video_tag)

    db.commit()

    return {"success": True, "vid": new_video.vid}


# 评论相关API
@app.get("/comment/{vid}/comments", response_model=List[schemas.Comment])
async def get_video_comments(vid: int, db: Session = Depends(get_db)):
    # 获取AI生成的视频摘要
    summary = db.query(models.VideoSummary).filter(models.VideoSummary.vid == vid).first()

    # 获取普通评论
    comments = db.query(models.Comment).filter(models.Comment.vid == vid).all()

    result = []

    # 如果有摘要，添加到结果的第一位
    if summary:
        result.append({
            "comment_id": 0,  # 特殊ID表示AI摘要
            "uid": 0,  # 特殊UID表示系统/AI
            "content": summary.summary,
            "likes_count": 0,
            "created_at": summary.created_at,
            "is_ai_summary": True,
            "user_avatar_url": "static/avatars/ai_avatar.png",
            "user_nickname": "AI助手"
        })

    # 添加普通评论
    for comment in comments:
        user = db.query(models.User).filter(models.User.uid == comment.uid).first()
        result.append({
            "comment_id": comment.comment_id,
            "uid": comment.uid,
            "content": comment.content,
            "likes_count": comment.likes_count,
            "created_at": comment.created_at,
            "is_ai_summary": False,
            "user_avatar_url": user.avatar_url if user else "default_avatar.png",
            "user_nickname": user.nickname if user else "未知用户"
        })

    return result


@app.post("/comment/{uid}/{vid}/post")
async def post_comment(
        uid: int,
        vid: int,
        text: str = Form(...),
        db: Session = Depends(get_db)
):
    # 检查用户是否存在
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 检查视频是否存在
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 创建评论
    comment = models.Comment(
        vid=vid,
        uid=uid,
        content=text
    )
    db.add(comment)

    # 更新视频评论计数
    video.comments_count += 1

    db.commit()
    db.refresh(comment)

    return {"success": True, "comment_id": comment.comment_id}


# Feed相关API
@app.get("/feed/{uid}")
async def get_feed(uid: int, request: Request, db: Session = Depends(get_db)):
    # 获取所有视频，允许重复
    videos = db.query(models.Video).limit(6).all()

    # 如果视频不足6个，重复填充
    if len(videos) < 6:
        while len(videos) < 6:
            additional_videos = db.query(models.Video).all()
            videos.extend(additional_videos)
            if len(additional_videos) == 0:  # 防止无限循环
                break
        videos = videos[:6]  # 确保只返回6个

    result = []
    for video in videos:
        # 获取作者信息
        author = db.query(models.User).filter(models.User.uid == video.uid).first()

        # 检查当前用户是否已点赞和收藏
        is_liked = db.query(models.Like).filter(
            models.Like.uid == uid,
            models.Like.vid == video.vid
        ).first() is not None

        is_favorited = db.query(models.Favorite).filter(
            models.Favorite.uid == uid,
            models.Favorite.vid == video.vid
        ).first() is not None

        # 获取视频可用的分辨率
        resolutions = db.query(models.VideoResolution).filter(models.VideoResolution.vid == video.vid).all()
        resolution_options = [
            {
                "resolution": r.resolution,
                "url": get_file_url(request, r.video_url, "video")
            }
            for r in resolutions
        ]

        result.append({
            "vid": video.vid,
            "author_uid": author.uid,
            "author_nickname": author.nickname,
            "author_avatar_url": get_avatar_url(request, author.avatar_url),
            "title": video.title,
            "description": video.description,
            "video_url": get_file_url(request, video.video_url, "video"),
            "cover_url": get_file_url(request, video.cover_url, "cover"),
            "likes_count": video.likes_count,
            "comments_count": video.comments_count,
            "favorites_count": video.favorites_count,
            "created_at": video.created_at,
            "is_liked": is_liked,
            "is_favorited": is_favorited,
            "resolution_options": resolution_options
        })

    return result


# 点赞、收藏、关注等交互API
@app.post("/video/{uid}/{vid}/like")
async def like_video(uid: int, vid: int, db: Session = Depends(get_db)):
    # 检查用户是否存在
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 检查视频是否存在
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 检查是否已点赞
    existing_like = db.query(models.Like).filter(
        models.Like.uid == uid,
        models.Like.vid == vid
    ).first()

    if existing_like:
        return {"success": False, "message": "Already liked"}

    # 创建点赞记录
    like = models.Like(uid=uid, vid=vid)
    db.add(like)

    # 更新视频点赞计数
    video.likes_count += 1

    db.commit()
    return {"success": True, "message": "Liked successfully"}


@app.post("/video/{uid}/{vid}/unlike")
async def unlike_video(uid: int, vid: int, db: Session = Depends(get_db)):
    # 查找点赞记录
    like = db.query(models.Like).filter(
        models.Like.uid == uid,
        models.Like.vid == vid
    ).first()

    if not like:
        return {"success": False, "message": "Not liked yet"}

    # 删除点赞记录
    db.delete(like)

    # 更新视频点赞计数
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if video and video.likes_count > 0:
        video.likes_count -= 1

    db.commit()
    return {"success": True, "message": "Unliked successfully"}


@app.post("/video/{uid}/{vid}/favorite")
async def favorite_video(uid: int, vid: int, db: Session = Depends(get_db)):
    # 检查用户是否存在
    user = db.query(models.User).filter(models.User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 检查视频是否存在
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 检查是否已收藏
    existing_favorite = db.query(models.Favorite).filter(
        models.Favorite.uid == uid,
        models.Favorite.vid == vid
    ).first()

    if existing_favorite:
        return {"success": False, "message": "Already favorited"}

    # 创建收藏记录
    favorite = models.Favorite(uid=uid, vid=vid)
    db.add(favorite)

    # 更新视频收藏计数
    video.favorites_count += 1

    db.commit()
    return {"success": True, "message": "Favorited successfully"}


@app.post("/video/{uid}/{vid}/unfavorite")
async def unfavorite_video(uid: int, vid: int, db: Session = Depends(get_db)):
    # 查找收藏记录
    favorite = db.query(models.Favorite).filter(
        models.Favorite.uid == uid,
        models.Favorite.vid == vid
    ).first()

    if not favorite:
        return {"success": False, "message": "Not favorited yet"}

    # 删除收藏记录
    db.delete(favorite)

    # 更新视频收藏计数
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if video and video.favorites_count > 0:
        video.favorites_count -= 1

    db.commit()
    return {"success": True, "message": "Unfavorited successfully"}

@app.post("/video/{uid}/{vid}/unlike")
async def unlike_video(uid: int, vid: int, db: Session = Depends(get_db)):
    # 查找点赞记录
    like = db.query(models.Like).filter(
        models.Like.uid == uid,
        models.Like.vid == vid
    ).first()

    if not like:
        return {"success": False, "message": "Not liked yet"}

    # 删除点赞记录
    db.delete(like)

    # 更新视频点赞计数
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if video and video.likes_count > 0:
        video.likes_count -= 1

    db.commit()
    return {"success": True}


@app.get("/video/{uid}/{vid}/isLike")
async def check_if_user_liked_video(uid: int, vid: int, db: Session = Depends(get_db)):
    """
    检查用户是否点赞了特定视频

    参数:
    - uid: 用户ID
    - vid: 视频ID

    返回:
    - is_liked: 布尔值，表示用户是否点赞了该视频
    """
    # 查询点赞记录
    like = db.query(models.Like).filter(
        models.Like.uid == uid,
        models.Like.vid == vid
    ).first()

    return {
        "uid": uid,
        "vid": vid,
        "is_liked": like is not None
    }


@app.get("/video/{uid}/{vid}/isFavorite")
async def check_if_user_favorited_video(uid: int, vid: int, db: Session = Depends(get_db)):
    """
    检查用户是否收藏了特定视频

    参数:
    - uid: 用户ID
    - vid: 视频ID

    返回:
    - is_favorited: 布尔值，表示用户是否收藏了该视频
    """
    # 查询收藏记录
    favorite = db.query(models.Favorite).filter(
        models.Favorite.uid == uid,
        models.Favorite.vid == vid
    ).first()

    return {
        "uid": uid,
        "vid": vid,
        "is_favorited": favorite is not None
    }


@app.get("/video/{uid}/{vid}/status")
async def check_video_status(uid: int, vid: int, db: Session = Depends(get_db)):
    """
    同时检查用户是否点赞和收藏了特定视频

    参数:
    - uid: 用户ID
    - vid: 视频ID

    返回:
    - is_liked: 布尔值，表示用户是否点赞了该视频
    - is_favorited: 布尔值，表示用户是否收藏了该视频
    """
    # 查询点赞记录
    like = db.query(models.Like).filter(
        models.Like.uid == uid,
        models.Like.vid == vid
    ).first()

    # 查询收藏记录
    favorite = db.query(models.Favorite).filter(
        models.Favorite.uid == uid,
        models.Favorite.vid == vid
    ).first()

    return {
        "uid": uid,
        "vid": vid,
        "is_liked": like is not None,
        "is_favorited": favorite is not None
    }



from video_processor import process_video_for_summary

@app.post("/video/{vid}/summarize")
async def summarize_video_content(
    vid: int,
    db: Session = Depends(get_db)
):
    """
    对指定视频进行大模型总结。
    如果视频时长和转录字数满足条件，则生成总结并存储。
    """
    video = db.query(models.Video).filter(models.Video.vid == vid).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # 检查是否已存在总结，避免重复处理
    existing_summary = db.query(models.VideoSummary).filter(models.VideoSummary.vid == vid).first()
    if existing_summary:
        return {"success": True, "message": "Video already summarized", "summary": existing_summary.summary}

    # 获取视频的完整文件路径
    # 注意：这里假设 video.video_url 存储的是相对于 'static/videos/' 的文件名
    video_full_path = os.path.join(os.getcwd(), "static", "videos", video.video_url)

    if not os.path.exists(video_full_path):
        raise HTTPException(status_code=404, detail="Video file not found on server.")

    # 异步执行总结任务
    summary = await process_video_for_summary(vid, video_full_path)

    if summary:
        # 将总结保存到数据库
        new_summary = models.VideoSummary(vid=vid, summary=summary)
        db.add(new_summary)
        db.commit()
        db.refresh(new_summary)
        return {"success": True, "message": "Video summarized successfully", "summary": summary}
    else:
        return {"success": False, "message": "Video did not meet summary criteria or encountered an error during processing."}
    

# 超分辨率相关API (替代方案)
@app.get("/video/{vid}/resolutions")
async def get_video_resolutions(vid: int, db: Session = Depends(get_db)):
    resolutions = db.query(models.VideoResolution).filter(models.VideoResolution.vid == vid).all()

    if not resolutions:
        raise HTTPException(status_code=404, detail="No resolution options found")

    return [{"resolution": r.resolution, "url": r.video_url} for r in resolutions]


from config import SUMMARY_MAX_VIDEOS_PER_BATCH

@app.post("/admin/trigger_summary_batch")
async def trigger_summary_batch(max_videos: int = None):
    """
    手动触发一批视频摘要任务（管理员接口）
    """
    try:
        # 如果没有指定max_videos，使用配置文件中的值
        max_videos = max_videos or SUMMARY_MAX_VIDEOS_PER_BATCH
        await video_summary_scheduler.run_summary_batch(max_videos_per_batch=max_videos)
        return {"success": True, "message": f"Summary batch triggered for up to {max_videos} videos"}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Failed to trigger summary batch")
    
@app.get("/admin/summary_status")
async def get_summary_status(db: Session = Depends(get_db)):
    """
    获取视频摘要状态统计（管理员接口）
    """
    try:
        total_videos = db.query(models.Video).count()
        videos_with_summary = db.query(models.VideoSummary).count()
        videos_without_summary = total_videos - videos_with_summary
        
        return {
            "total_videos": total_videos,
            "videos_with_summary": videos_with_summary,
            "videos_without_summary": videos_without_summary,
            "summary_coverage": f"{(videos_with_summary/total_videos*100):.1f}%" if total_videos > 0 else "0%"
        }
    except Exception as e:

        raise HTTPException(status_code=500, detail="Failed to get summary status")
    
# 启动服务器
if __name__ == "__main__":
    import uvicorn

    # 确保目录存在
    os.makedirs("static/videos", exist_ok=True)
    os.makedirs("static/covers", exist_ok=True)
    os.makedirs("static/avatars", exist_ok=True)
    os.makedirs("static/videos/resolutions", exist_ok=True)

    # 创建数据库表
    models.Base.metadata.create_all(bind=engine)

    uvicorn.run(app, host="0.0.0.0", port=8000)