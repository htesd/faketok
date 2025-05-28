# models.py (SQLAlchemy models)
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    uid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)
    nickname = Column(String(50), nullable=False)
    bio = Column(Text)
    avatar_url = Column(String(255), default="default_avatar.png")
    created_at = Column(DateTime, default=func.now())


class Video(Base):
    __tablename__ = "videos"

    vid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uid = Column(Integer, ForeignKey("users.uid"), nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(Text)
    video_url = Column(String(255), nullable=False)
    cover_url = Column(String(255), nullable=False)
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    favorites_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


class Tag(Base):
    __tablename__ = "tags"

    tag_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)


class VideoTag(Base):
    __tablename__ = "video_tags"

    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.tag_id"), primary_key=True)


class Comment(Base):
    __tablename__ = "comments"

    comment_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vid = Column(Integer, ForeignKey("videos.vid"), nullable=False)
    uid = Column(Integer, ForeignKey("users.uid"), nullable=False)
    content = Column(Text, nullable=False)
    likes_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())


class Like(Base):
    __tablename__ = "likes"

    uid = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())


class Favorite(Base):
    __tablename__ = "favorites"

    uid = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())


class Follow(Base):
    __tablename__ = "follows"

    follower_id = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    followed_id = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())


class VideoSummary(Base):
    __tablename__ = "video_summaries"

    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())


class VideoResolution(Base):
    __tablename__ = "video_resolutions"

    resolution_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vid = Column(Integer, ForeignKey("videos.vid"), nullable=False)
    resolution = Column(String(20), nullable=False)
    video_url = Column(String(255), nullable=False)# models.py (SQLAlchemy models)
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    uid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)
    nickname = Column(String(50), nullable=False)
    bio = Column(Text)
    avatar_url = Column(String(255), default="default_avatar.png")
    created_at = Column(DateTime, default=func.now())

class Video(Base):
    __tablename__ = "videos"

    vid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    uid = Column(Integer, ForeignKey("users.uid"), nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(Text)
    video_url = Column(String(255), nullable=False)
    cover_url = Column(String(255), nullable=False)
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    favorites_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

class Tag(Base):
    __tablename__ = "tags"

    tag_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)

class VideoTag(Base):
    __tablename__ = "video_tags"

    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.tag_id"), primary_key=True)

class Comment(Base):
    __tablename__ = "comments"

    comment_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vid = Column(Integer, ForeignKey("videos.vid"), nullable=False)
    uid = Column(Integer, ForeignKey("users.uid"), nullable=False)
    content = Column(Text, nullable=False)
    likes_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

class Like(Base):
    __tablename__ = "likes"

    uid = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())

class Favorite(Base):
    __tablename__ = "favorites"

    uid = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())

class Follow(Base):
    __tablename__ = "follows"

    follower_id = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    followed_id = Column(Integer, ForeignKey("users.uid"), primary_key=True)
    created_at = Column(DateTime, default=func.now())

class VideoSummary(Base):
    __tablename__ = "video_summaries"

    vid = Column(Integer, ForeignKey("videos.vid"), primary_key=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())

class VideoResolution(Base):
    __tablename__ = "video_resolutions"

    resolution_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vid = Column(Integer, ForeignKey("videos.vid"), nullable=False)
    resolution = Column(String(20), nullable=False)
    video_url = Column(String(255), nullable=False)