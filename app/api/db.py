from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/basketball")


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_path: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def upsert_job(job_id: str, mode: str, status: str, video_path: str, result_path: str | None = None) -> None:
    with SessionLocal() as session:
        _upsert_job(session, job_id, mode, status, video_path, result_path)


def _upsert_job(
    session: Session, job_id: str, mode: str, status: str, video_path: str, result_path: str | None = None
) -> None:
    now = datetime.utcnow()
    row = session.get(JobRecord, job_id)
    if row is None:
        row = JobRecord(
            job_id=job_id,
            mode=mode,
            status=status,
            video_path=video_path,
            result_path=result_path,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
    else:
        row.mode = mode
        row.status = status
        row.video_path = video_path
        row.result_path = result_path
        row.updated_at = now
    session.commit()


def get_job_row(job_id: str) -> JobRecord | None:
    with SessionLocal() as session:
        return session.get(JobRecord, job_id)


def get_job_status_counts() -> dict[str, int]:
    with SessionLocal() as session:
        rows = session.query(JobRecord.status, func.count(JobRecord.job_id)).group_by(JobRecord.status).all()
        return {status: int(count) for status, count in rows}
