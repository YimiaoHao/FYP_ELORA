from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 数据库文件：项目根目录下的 elora.db
DATABASE_URL = "sqlite:///./elora.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # 允许多线程访问
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
