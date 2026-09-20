import streamlit as st
import requests


API_BASE_URL = "http://127.0.0.1:8001"


st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="🤖",
    layout="centered",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None


# --------------------------------------------------
# Login screen
# --------------------------------------------------

if not st.session_state.access_token:

    st.title("🤖 Production RAG Assistant")
    st.write("Login to access the RAG system.")

    username = st.text_input(
        "Username",
        placeholder="Enter your username",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
    )

    if st.button("Login"):

        if not username.strip() or not password:
            st.warning("Please enter your username and password.")

        else:
            try:
                response = requests.post(
                    f"{API_BASE_URL}/login",
                    json={
                        "username": username,
                        "password": password,
                    },
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state.access_token = result[
                        "access_token"
                    ]

                    st.session_state.username = result[
                        "username"
                    ]

                    st.session_state.role = result[
                        "role"
                    ]

                    st.rerun()

                else:
                    st.error(
                        f"Login failed: {response.json().get('detail', 'Unknown error')}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running on port 8001."
                )


# --------------------------------------------------
# Authenticated application
# --------------------------------------------------

else:

    st.title("🤖 Production RAG Assistant")

    st.write(
        f"Logged in as **{st.session_state.username}** "
        f"({st.session_state.role})"
    )

    if st.button("Logout"):
        st.session_state.access_token = None
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()

    st.divider()

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is 24-karat gold?",
    )

    if st.button("Ask"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    headers={
                        "Authorization": (
                            f"Bearer {st.session_state.access_token}"
                        )
                    },
                    json={
                        "question": question,
                    },
                )

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Sources")

                    if result["sources"]:

                        for source in result["sources"]:
                            st.write(
                                f"**{source['filename']}** — "
                                f"Page {source['page']} — "
                                f"{source['section']}"
                            )

                    else:
                        st.info("No accessible sources found.")

                elif response.status_code == 401:

                    st.error(
                        "Your session has expired. Please log in again."
                    )

                    st.session_state.access_token = None
                    st.session_state.username = None
                    st.session_state.role = None

                else:

                    st.error(
                        f"API Error {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running on port 8001."
                )