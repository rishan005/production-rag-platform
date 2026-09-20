from typing import Optional

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)

    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)

    password_hash: str

    role: str = Field(default="guest")