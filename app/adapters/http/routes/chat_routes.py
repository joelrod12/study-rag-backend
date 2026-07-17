from fastapi import APIRouter
from pydantic import BaseModel

from app.application.use_cases.ask_question_use_case import (
    AskQuestionUseCase,
)
from app.infrastructure.ai.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.infrastructure.ai.llm.groq_provider import GroqProvider
from app.infrastructure.ai.rag.prompt_builder import PromptBuilder
from app.infrastructure.ai.rag.retriever import Retriever
from app.infrastructure.vector_store.chroma_vector_store import (
    ChromaVectorStore,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(
    request: QuestionRequest,
):
    """
    Answers a question using the indexed documents.
    """

    vector_store = ChromaVectorStore()

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    prompt_builder = PromptBuilder()

    llm = GroqProvider()

    use_case = AskQuestionUseCase(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    answer = use_case.execute(
        request.question,
    )

    return {
        "question": request.question,
        "answer": answer,
    }
