from app.pipeline import prepare_pipeline
from app.rag import RAGPipeline
from app.retrival import semantic_search


# -------------------------------------------------
# PREPARE RAG PIPELINE
# -------------------------------------------------

chunks, bm25_index = prepare_pipeline()


# -------------------------------------------------
# CREATE RAG SYSTEM
# -------------------------------------------------

rag = RAGPipeline(
    chunks=chunks,
    bm25_index=bm25_index,
)


# -------------------------------------------------
# ASK QUESTION
# -------------------------------------------------

query = "How pure is 24-karat gold?"

answer = rag.answer(query)


# -------------------------------------------------
# FINAL ANSWER
# -------------------------------------------------

print("\n========== FINAL ANSWER ==========\n")

print(answer["answer"])


# -------------------------------------------------
# SOURCES
# -------------------------------------------------

print("\n========== SOURCES ==========\n")

for source in answer["sources"]:

    print(
        f"{source['filename']} | "
        f"Page {source['page']} | "
        f"{source['section']} | "
        f"{source['chunk_id']}"
    )


# -------------------------------------------------
# CITATION VERIFICATION
# -------------------------------------------------

print("\n========== CITATION VERIFICATION ==========\n")

citation_check = answer["citation_verification"]

print(
    f"Supported : "
    f"{citation_check['supported']}"
)

print(
    f"Sources   : "
    f"{citation_check['source_ids']}"
)

print(
    f"Reason    : "
    f"{citation_check['reason']}"
)


# -------------------------------------------------
# RETRIEVAL INFORMATION
# -------------------------------------------------

print("\n========== RETRIEVAL ==========\n")

retrieval = answer["retrieval"]

print(
    f"Method      : {retrieval['method']}"
)

print(
    f"Retrieval K : {retrieval['retrieval_k']}"
)

print(
    f"Rerank K    : {retrieval['rerank_k']}"
)


# -------------------------------------------------
# TOKEN USAGE
# -------------------------------------------------

print("\n========== TOKEN USAGE ==========\n")

print(
    f"Input tokens  : "
    f"{answer['input_tokens']}"
)

print(
    f"Output tokens : "
    f"{answer['output_tokens']}"
)
print("\n========== PERMISSION TEST ==========\n")

rag.memory.clear()

public_results = semantic_search(
    "How pure is 24-karat gold?",
    access_level="public",
)

print("Public user results:")
for result in public_results:
    print(
        result["chunk_id"],
        "|",
        result["access_level"]
    )

print("\n========== EMPLOYEE PERMISSION TEST ==========\n")

employee_query = "What is the maximum annual performance bonus?"

# Public user
rag.memory.clear()

public_employee_results = rag.answer(
    employee_query,
    access_level="public",
)

print("PUBLIC USER:")
for source in public_employee_results["sources"]:
    print(source["filename"], "|", source["chunk_id"])


# Employee user
rag.memory.clear()

employee_results = rag.answer(
    employee_query,
    access_level="employee",
)

print("\nEMPLOYEE USER:")
for source in employee_results["sources"]:
    print(source["filename"], "|", source["chunk_id"])


# Admin user
rag.memory.clear()

admin_results = rag.answer(
    employee_query,
    access_level="admin",
)

print("\nADMIN USER:")
for source in admin_results["sources"]:
    print(source["filename"], "|", source["chunk_id"])

# ==================================================
# FINAL STEP 18 VALIDATION
# ==================================================

print("\n========== FINAL VALIDATION ==========\n")

# --------------------------------------------------
# TEST 1: PUBLIC USER + PUBLIC QUESTION
# --------------------------------------------------

print("TEST 1: Public user asks public question")

rag.memory.clear()

public_gold = rag.answer(
    "How pure is 24-karat gold?",
    access_level="public",
)

print("Answer:", public_gold["answer"])
print("Sources:", public_gold["sources"])


# --------------------------------------------------
# TEST 2: PUBLIC USER + EMPLOYEE QUESTION
# --------------------------------------------------

print("\nTEST 2: Public user asks employee-only question")

rag.memory.clear()

public_hr = rag.answer(
    "What is the maximum annual performance bonus?",
    access_level="public",
)

print("Answer:", public_hr["answer"])
print("Sources:", public_hr["sources"])


# --------------------------------------------------
# TEST 3: EMPLOYEE USER + EMPLOYEE QUESTION
# --------------------------------------------------

print("\nTEST 3: Employee asks employee-only question")

employee_hr = rag.answer(
    "What is the maximum annual performance bonus?",
    access_level="employee",
)

print("Answer:", employee_hr["answer"])
print("Sources:", employee_hr["sources"])


# --------------------------------------------------
# TEST 4: ADMIN + EMPLOYEE QUESTION
# --------------------------------------------------

print("\nTEST 4: Admin asks employee-only question")

admin_hr = rag.answer(
    "What is the maximum annual performance bonus?",
    access_level="admin",
)

print("Answer:", admin_hr["answer"])
print("Sources:", admin_hr["sources"])


# --------------------------------------------------
# TEST 5: IRRELEVANT QUESTION
# --------------------------------------------------

print("\nTEST 5: Irrelevant question")

irrelevant = rag.answer(
    "What is the capital of Japan?",
    access_level="public",
)

print("Answer:", irrelevant["answer"])
print("Sources:", irrelevant["sources"])

# ==================================================
# STEP 19 - CONVERSATION MEMORY TEST
# ==================================================

print("\n========== STEP 19 MEMORY TEST ==========\n")

# Start a fresh conversation
rag.memory.clear()

# Question 1
question_1 = "What is 24-karat gold?"

result_1 = rag.answer(
    question_1,
    access_level="public",
)

print("QUESTION 1:")
print(question_1)

print("\nANSWER 1:")
print(result_1["answer"])


# Question 2 - follow-up question
question_2 = "What is its atomic number?"

result_2 = rag.answer(
    question_2,
    access_level="public",
)

print("\nQUESTION 2:")
print(question_2)

print("\nANSWER 2:")
print(result_2["answer"])


# Show stored conversation
print("\nMEMORY:")
for message in rag.memory.get_history():
    print(message["role"], ":", message["content"])