"""
RAG Evaluation Script using RAGAS.

Runs a set of test questions through the full RAG pipeline,
then scores the results using:
  - faithfulness:      Did the LLM answer based on the retrieved chunks? (detects hallucinations)
  - answer_relevancy:  Does the answer actually address the question asked?

"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.rag_chain import build_rag

TEST_QUESTIONS = [
    "What is the main research question of this thesis?",
    "What methodology was used in this study?",
    "What are the main findings of the research?",
    "What datasets were used in the experiments?",
    "What are the limitations of this study?",
]

MAX_CHUNK_CHARS = 1500


def run_evaluation():
    api_key = os.getenv("OPENAI_API_KEY")

    # Configure the LLM and embeddings for RAGAS evaluation
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))

    # Inject our LLM + embeddings into the metrics
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = emb

    questions = []
    answers = []
    contexts = []

    print(f"\nRunning {len(TEST_QUESTIONS)} questions through the RAG pipeline...\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question}")
        answer, _, chunk_texts = build_rag(question)
        questions.append(question)
        answers.append(answer)
        # Limit to 3 chunks and truncate to avoid max_tokens errors in RAGAS
        contexts.append([c[:MAX_CHUNK_CHARS] for c in chunk_texts[:3]])

    # Build the RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    # Use to_pandas() to get per-question scores and compute averages
    df = results.to_pandas()

    for idx, row in df.iterrows():
        print(f"  Q{idx+1}: {row['user_input'][:60]}...")
        print(f"      faithfulness: {row['faithfulness']:.4f}   answer_relevancy: {row['answer_relevancy']:.4f}")
        print()

    avg_faith = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()

    print(f"  faithfulness      : {avg_faith:.4f}  (0=hallucinating, 1=grounded)")
    print(f"  answer_relevancy  : {avg_relevancy:.4f}  (0=off-topic, 1=relevant)")
    return results


if __name__ == "__main__":
    run_evaluation()