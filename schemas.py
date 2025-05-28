# schemas.py (Pydantic models)
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserInfo(BaseModel):
    uid: int
    nickname: str
    bio: Optional[str] = None
    avatar_url: str
    videos: List[int]
    favorites: List[int]
    likes: List[int]
    following: List[int]
    followers: List[int]

class VideoPreviewInfo(BaseModel):
    vid: int
    cover_url: str
    likes_count: int

class ResolutionOption(BaseModel):
    resolution: str
    url: str

class VideoInfo(BaseModel):
    vid: int
    author_uid: int
    author_nickname: str
    author_avatar_url: str
    title: str
    description: Optional[str] = None
    video_url: str
    cover_url: str
    likes_count: int
    comments_count: int
    favorites_count: int
    created_at: datetime
    resolution_options: Optional[List[ResolutionOption]] = None

class Comment(BaseModel):
    comment_id: int
    uid: int
    content: str
    likes_count: int
    created_at: datetime
    is_ai_summary: bool = False
    user_avatar_url: str
    user_nickname: str# schemas.py (Pydantic models)
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserInfo(BaseModel):
    uid: int
    nickname: str
    bio: Optional[str] = None
    avatar_url: str
    videos: List[int]
    favorites: List[int]
    likes: List[int]
    following: List[int]
    followers: List[int]

class VideoPreviewInfo(BaseModel):
    vid: int
    cover_url: str
    likes_count: int

class ResolutionOption(BaseModel):
    resolution: str
    url: str

class VideoInfo(BaseModel):
    vid: int
    author_uid: int
    author_nickname: str
    author_avatar_url: str
    title: str
    description: Optional[str] = None
    video_url: str
    cover_url: str
    likes_count: int
    comments_count: int
    favorites_count: int
    created_at: datetime
    resolution_options: Optional[List[ResolutionOption]] = None

class Comment(BaseModel):
    comment_id: int
    uid: int
    content: str
    likes_count: int
    created_at: datetime
    is_ai_summary: bool = False
    user_avatar_url: str
    user_nickname: str