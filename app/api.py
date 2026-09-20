from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.pipeline import prepare_pipeline
from app.rag import RAGPipeline
from fastapi import HTTPException
from app.database import create_db_and_tables
from sqlmodel import Session, select

from app.auth import hash_password , verify_password, create_access_token
from app.models import User
from app.database import engine
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import decode_access_token

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials

    try:
        payload = decode_access_token(token)

        username = payload.get("sub")
        role = payload.get("role")

        if not username or not role:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token.",
            )

        return {
            "username": username,
            "role": role,
        }

    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

def require_admin(
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required.",
        )

    return current_user

app = FastAPI(
    title="Production RAG Platform",
    description="Production-style Retrieval-Augmented Generation API",
    version="1.0.0",
)


# Prepare the RAG system when the API starts
chunks, bm25_index = prepare_pipeline()
rag = RAGPipeline(chunks, bm25_index)


@app.get("/")
def root():
    return {
        "message": "Production RAG API is running"
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    email: str
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(request: RegisterRequest):
    with Session(engine) as session:

        # Check whether username already exists
        existing_username = session.exec(
            select(User).where(User.username == request.username)
        ).first()

        if existing_username:
            raise HTTPException(
                status_code=400,
                detail="Username already exists.",
            )

        # Check whether email already exists
        existing_email = session.exec(
            select(User).where(User.email == request.email)
        ).first()

        if existing_email:
            raise HTTPException(
                status_code=400,
                detail="Email already exists.",
            )

        # Create new user
        user = User(
            username=request.username,
            email=request.email,
            password_hash=hash_password(request.password),
            role="guest",
        )

        session.add(user)
        session.commit()
        session.refresh(user)

        return {
            "message": "User registered successfully.",
            "username": user.username,
            "role": user.role,
        }

@app.post("/login")
def login(request: LoginRequest):
    with Session(engine) as session:

        # Find user by username
        user = session.exec(
            select(User).where(User.username == request.username)
        ).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )

        # Verify password
        if not verify_password(
            request.password,
            user.password_hash,
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )

        # Create JWT using the user's server-side role
        access_token = create_access_token(
            username=user.username,
            role=user.role,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username,
            "role": user.role,
        }

@app.patch("/admin/users/{username}/role")
def update_user_role(
    username: str,
    new_role: str,
    current_user: dict = Depends(require_admin),
):
    allowed_roles = {"guest", "employee", "admin"}

    if new_role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail="Invalid role. Allowed roles: guest, employee, admin.",
        )

    with Session(engine) as session:

        user = session.exec(
            select(User).where(User.username == username)
        ).first()

        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        user.role = new_role

        session.add(user)
        session.commit()
        session.refresh(user)

        return {
            "message": "User role updated successfully.",
            "username": user.username,
            "role": user.role,
        }


@app.post("/ask")
def ask(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        result = rag.answer(
            query=request.question,
            access_level=(
                "public"
                if current_user["role"] == "guest"
                else current_user["role"]
            ),
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )