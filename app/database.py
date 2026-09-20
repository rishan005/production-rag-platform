from sqlmodel import SQLModel, create_engine
from app.config import CONFIG
from app.models import User


DATABASE_URL = CONFIG.database_url


engine = create_engine(
    DATABASE_URL,
    echo=False,
)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    with engine.connect() as connection:
        print("PostgreSQL connection successful!")