from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import os

def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3 )

def build_chain(system_prompt : str) :
    llm = get_llm()

    return (
         RunnableLambda(lambda x: {"text" : x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human" , "{text}")
    ]) | llm | StrOutputParser()
    )

def extract_action_items(transcript : str) -> str:
    chain = build_chain(
       "You are an expert analyst. From the transcript of a video or meeting, "
        "extract action items: tasks, next steps, or practical recommendations "
        "that someone should act on. For each item give:\n"
        "- The task\n"
        "- Owner (only if a person is named, else omit)\n"
        "- Deadline (only if mentioned, else omit)\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert analyst. From the transcript of a video or meeting, "
        "extract key decisions, conclusions, or firm positions stated by the speakers. "
        "Format as a numbered list. If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert analyst. From the transcript of a video or meeting, "
        "extract unresolved questions, open issues, or topics needing follow-up. "
        "Format as a numbered list. If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)