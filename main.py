import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama


# TODO: Include memory
def __setup_chain() -> RunnableSerializable:
    prompt_template = PromptTemplate.from_template(
        template="""
        Eres un experto en sistemas de informacion, transformacion digital y tecnologia.
        Debes responder las preguntas de manera sencilla y no tecnica, las personas que estan
        interactuando contigo no tienen experiencia en el campo de tecnologia.

        Pregunta: {question}
        """
    )
    llm = ChatOllama(model="llama3.2", temperature=0)

    # Create chain
    qa_chain = prompt_template | llm | StrOutputParser()
    return qa_chain


def ask_question(question: str, history) -> str:
    qa_chain = __setup_chain()

    try:
        response = qa_chain.invoke(question)
        return response
    except Exception as e:
        return f"Error: {e}"


def main():
    gr.ChatInterface(
        fn=ask_question,
        type="messages",
        chatbot=gr.Chatbot(
            type="messages",
            avatar_images=("assets/knight-logo.jpg", "assets/mygopher.png"),
        ),
    ).launch(share=False)


if __name__ == "__main__":
    main()
