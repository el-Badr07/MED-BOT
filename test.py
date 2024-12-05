import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Initialize the LLM model (ChatOllama)
llm = ChatOllama(model='smollm2:1.7b')

# Define the prompt template for asking questions based on the JSON document
prompt = PromptTemplate(
    input_variables=["json_data", "question"],
    template="""
You are an assistant that can answer questions based on structured data in JSON format.

Here is the provided data:
{json_data}

Now, please answer the following question based on the data:
{question}
"""
)

# Function to load the JSON document
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)  # Load and return the JSON data as a Python dictionary

# Create the LLM chain with the prompt template
chain = LLMChain(llm=llm, prompt=prompt)

# Load the saved JSON document
json_file_path = "TP_RAG\output_report.json"
json_data = load_json(json_file_path)

# Ask a specific question based on the JSON document
question = "interpret the data and explain it and finaly give some diagnosis?"

# Run the chain with the JSON data and the question
response = chain.invoke({"json_data": json.dumps(json_data), "question": question})

# Print the response
print("Answer from the model:", response['text'])
