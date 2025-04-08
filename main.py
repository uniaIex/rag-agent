from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retrieval
#specify model
model = OllamaLLM(model="llama3.2")

#prompt

template = """
You are an expert in answering questions about a pizza restaurant

Here are some reviews: {reviews}

Here is the question to answear: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

#create a chain, which is a combination of the prompt and the model
# The chain will take the prompt and the model and combine them
# to create a single object that can be used to generate text
chain = prompt | model

# The chain will take the prompt and the model and combine them
while True:
    print("\n\n------------------------------------------")
    question = input("Enter your question, press q to quit: ")
    print("\n\n------------------------------------------")
    if question == "q":
        break
    reviews = retrieval.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)