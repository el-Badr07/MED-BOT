from unstract.llmwhisperer import LLMWhispererClientV2
from langchain_community.chat_models import ChatOllama
import json
import re
from langchain_core.output_parsers import JsonOutputParser

#client = LLMWhispererClientV2()
# Provide the base URL and API key explicitly
#Return the result as a valid JSON object without any additional text or comments.

def interpret_json(file):
    client = LLMWhispererClientV2(base_url="https://llmwhisperer-api.us-central.unstract.com/api/v2", api_key="FdutG4XNpnK5ILGwYTei2WWhhdnSEan-oMurX2jVUEE")
    def llmwhisper(file_path1):
        whisper = client.whisper(
            file_path=file_path1, 
            wait_for_completion=True,
            wait_timeout=200
        )
        return whisper['extraction']['result_text']


    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # Initialize the LLM
    llm = ChatOllama(model='llama3.2')

    # Define the prompt
    prompt = PromptTemplate(
        input_variables=["report_text"],
        template="""
    You are a medical data parser. Convert the following medical report into a structured JSON format. 

    Report Text:
    {report_text}

    Ensure the JSON is well-formatted and includes all information from the report.

    Output only json no text.

    """
    )

    # Create the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt,output_parser=JsonOutputParser())

    # Input text (replace with your report)
    #report_text = [llmwhisper('TP_RAG\documents\cbc-report-format.pdf')]

    #file_path = 'TP_RAG/documents/cbc-report-format.pdf'

    # Extract text using Whisperer
    extracted_report_text = llmwhisper(file)

    # Run the chain with the extracted report text
    response = chain.invoke({"report_text": extracted_report_text})

    llm = ChatOllama(model='llama3.2')

    # Define the prompt template for asking questions based on the JSON document
    prompt = PromptTemplate(
        input_variables=["json_data", "question"],
        template="""
    You are an experienced and compassionate medical professional that can answer questions based on structured data in JSON format. You are great at answering medical questions, explaining symptoms, treatments, and diagnoses in a clear and empathetic way. 

    Here is the provided data:
    {json_data}

    Now, please answer the following question based on the data:
    {question}
    """
    )



    # Create the LLM chain with the prompt template
    chain = LLMChain(llm=llm, prompt=prompt)

    # Load the saved JSON document



    # Ask a specific question based on the JSON document
    question = "Based on the provided medical report data in JSON format, analyze and interpret the details of the patient's medical information, including any relevant medical history, test results, and observations. Summarize the key findings, explain their significance, and provide a possible diagnosis or a set of differential diagnoses based on standard medical knowledge. Include any recommendations for further tests or treatments if applicable."



    # Run the chain with the JSON data and the question
    response1 = chain.invoke({"json_data": response, "question": question})

    # Print the response
    #print("Answer from the model:", response1['text'])
    return response1['text']






'''
try:
    # Assuming the response is already a dictionary (parsed JSON)
    with open("output_report.json", "w") as json_file:
        json.dump(response, json_file, indent=4)
    print("JSON saved successfully.")
except Exception as e:
    print(f"Error saving JSON: {e}")
'''
