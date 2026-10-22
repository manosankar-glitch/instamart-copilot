
from pathlib import Path
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10

chunks = []
vectorizer = None
chunk_matrix = None


def load_policy_chunks(overlap=DEFAULT_OVERLAP):
    """Split each policy into 40-word chunks with the requested overlap."""
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("overlap must be between 0 and 39 words")

    all_chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        step = CHUNK_SIZE - overlap

        for start in range(0, len(words), step):
            end = min(start + CHUNK_SIZE, len(words))
            all_chunks.append({
                "source": path.name,
                "text": " ".join(words[start:end]),
            })

            # Stop as soon as this chunk reaches the document's last word.
            if end >= len(words):
                break

    return all_chunks


def build_index(overlap=DEFAULT_OVERLAP):
    """Build and fit a TF-IDF index for the selected chunk overlap."""
    global chunks, vectorizer, chunk_matrix

    chunks = load_policy_chunks(overlap)
    if not chunks:
        raise RuntimeError(f"No policy .txt files found in {POLICY_DIR}")

    vectorizer = TfidfVectorizer()
    chunk_matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )
    return chunks


def retrieve(query, k=3):
    """Return the k most similar policy chunks, ranked by cosine similarity."""
    if vectorizer is None or chunk_matrix is None:
        build_index()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, chunk_matrix).ravel()

    # Stable tie-breaking: higher score first, then original chunk order.
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], i),
    )[:k]

    return [
        {
            "source": chunks[i]["source"],
            "score": float(scores[i]),
            "text": chunks[i]["text"],
        }
        for i in ranked_indices
    ]


def answer(question):
    """Answer using retrieved context and return the text and sources."""
    retrieved = retrieve(question, k=3)

    context_lines = ["CONTEXT:"]
    for chunk in retrieved:
        context_lines.append(f"[{chunk['source']}]")
        context_lines.append(chunk["text"])
    context_lines.extend(["QUESTION:", question])
    user_content = "\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only from the supplied policy "
                "context. If the context does not contain the answer, reply "
                "exactly: I could not find that information in the policy documents."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    response = mock_llm.chat(messages)
    text = response.choices[0].message.content or ""

    return {
        "answer": text,
        "sources": [chunk["source"] for chunk in retrieved],
    }


def evaluate_current_index():
    """Evaluate source retrieval, expected facts, and missing-policy fallback."""
    with EVAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    answerable = 0
    source_hits = 0
    fact_hits = 0
    unanswerable = 0
    fallback_hits = 0
    failures = []

    for row in rows:
        question = row["question"]
        expected_doc = (row.get("expected_doc") or "").strip()
        expected_fact = (row.get("expected_fact") or "").strip()

        retrieved = retrieve(question, k=3)
        result = answer(question)
        response_text = result["answer"]

        is_answerable = (
            expected_doc.lower() not in {"", "none", "n/a", "na"}
            and expected_fact != ""
        )

        if is_answerable:
            answerable += 1
            has_source = any(
                item["source"] == expected_doc for item in retrieved
            )
            has_fact = expected_fact in response_text

            source_hits += int(has_source)
            fact_hits += int(has_fact)

            if not has_source or not has_fact:
                failures.append({
                    "question": question,
                    "expected_doc": expected_doc,
                    "expected_fact": expected_fact,
                    "retrieved_sources": [x["source"] for x in retrieved],
                    "answer": response_text,
                })
        else:
            unanswerable += 1
            is_fallback = response_text.startswith("I could not find")
            fallback_hits += int(is_fallback)

            if not is_fallback:
                failures.append({
                    "question": question,
                    "expected_doc": expected_doc,
                    "expected_fact": expected_fact,
                    "retrieved_sources": [x["source"] for x in retrieved],
                    "answer": response_text,
                })

    return {
        "total_questions": len(rows),
        "answerable": answerable,
        "source_hits": source_hits,
        "fact_hits": fact_hits,
        "unanswerable": unanswerable,
        "fallback_hits": fallback_hits,
        "failures": failures,
    }


def print_evaluation(label, results):
    print(f"\n--- {label} ---")
    print(f"Total questions: {results['total_questions']}")
    print(
        "Answerable questions with expected document in top 3: "
        f"{results['source_hits']}/{results['answerable']}"
    )
    print(
        "Answerable questions containing expected fact: "
        f"{results['fact_hits']}/{results['answerable']}"
    )
    print(
        "Unanswerable questions with correct fallback: "
        f"{results['fallback_hits']}/{results['unanswerable']}"
    )

    if results["failures"]:
        print("\nFailures:")
        for failure in results["failures"]:
            print(f"- Question: {failure['question']}")
            print(f"  Expected document: {failure['expected_doc']}")
            print(f"  Expected fact: {failure['expected_fact']}")
            print(f"  Retrieved: {failure['retrieved_sources']}")
            print(f"  Answer: {failure['answer']}")


if __name__ == "__main__":
    build_index(overlap=10)

    print("--- CHUNK COUNTS WITH 10-WORD OVERLAP ---")
    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")
    print(f"Total chunks: {len(chunks)}")

    print("\n--- RETRIEVAL TEST ---")
    query = "refund policy late delivery"
    for rank, item in enumerate(retrieve(query, k=3), start=1):
        print(
            f"{rank}. {item['source']} | "
            f"score={item['score']:.4f} | {item['text']}"
        )

    print("\n--- ANSWER TEST ---")
    test_question = "What delivery time makes a completed order eligible for a full refund?"
    print(answer(test_question))

    # Evaluate using the required 10-word overlap.
    build_index(overlap=10)
    overlap_results = evaluate_current_index()
    print_evaluation("EVALUATION: 40 WORDS, 10-WORD OVERLAP", overlap_results)

    # Rebuild the index and repeat the entire evaluation without overlap.
    build_index(overlap=0)
    no_overlap_results = evaluate_current_index()
    print_evaluation("EVALUATION: 40 WORDS, ZERO OVERLAP", no_overlap_results)

    print("\n--- TF-IDF EXPLANATION ---")
    print(
        "Term frequency (TF) measures how often a term appears in a chunk; "
        "inverse document frequency (IDF) gives less weight to terms that "
        "appear across many chunks. TF-IDF can miss semantic matches when "
        "a question and a policy use different words to express the same idea."
    )
