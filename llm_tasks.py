import pandas as pd  
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re




df=pd.read_csv("data.csv")
print(df.head())

# print(df.columns)
# print(df.shape)
# df=df.iloc[:10]
# print(df.shape)






model_name="gemma3:27b"
#model_name="mistral-small3.1"
#model_name="deepseek-r1:32b"


filename= "data_"+model_name+".csv"



chat = OllamaLLM(model=model_name)



template_string1 = """Translate in English the text in that is delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation. It is mandatory to write only in English.)
"""

template_string2 = """Make a journal title in English of the text delimited by triple backticks 
text: ```{text}``` (I expressly want only one title with one sentence,
 I expressly don't want any comment and I don't want to see the keyword "title" and I don't want to see some "*". 
 It is mandatory to write only 1 title of 1 sentence in English. I expressely don't want to see many empty lines. 
 I don't want any explanation. please answer with 1 title only).
"""

template_string3 = """Answer the question delimited by triple backticks 
text: ```{question}``` in 3 sentences (max) in English according to the text delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation. I expressly want only 3 sentences 
and I don't want to see some "*". I expressely want complete sentences. It is mandatory to write only in English.)
"""



################## TRANSLATE
answer=[]

for t in df["text"]:
    
    prompt_template = ChatPromptTemplate.from_template(template_string1)

    ##Lets plugin our constants from the template
    customer_messages = prompt_template.format_messages(text=t)

    # customer_messages = prompt_template.format_messages(
    #                     style=_style,
    #                     text=_message)

    #print(customer_messages)

    ##Let's get the chat response from the customer messages
    customer_response = chat.invoke(customer_messages)

    


    customer_response = re.sub(r"<think>.*?</think>\n?", "",customer_response, flags=re.DOTALL)


    print("ANNNNNNNNNNNNSSSSSSSSSSSSWWWWWWWWER translate",customer_response)
    answer.append(customer_response)

df["translate_"+model_name]=answer


################## SUMMARY
answer=[]

for t in df["text"]:
    
    prompt_template = ChatPromptTemplate.from_template(template_string2)

    ##Lets plugin our constants from the template
    customer_messages = prompt_template.format_messages(text=t)

    # customer_messages = prompt_template.format_messages(
    #                     style=_style,
    #                     text=_message)

    #print(customer_messages)

    ##Let's get the chat response from the customer messages
    customer_response = chat.invoke(customer_messages)

    customer_response = re.sub(r"<think>.*?</think>\n?", "",customer_response, flags=re.DOTALL)

    print("ANNNNNNNNNNNNSSSSSSSSSSSSWWWWWWWWER SUMMARY",customer_response)
    answer.append(customer_response)

df["summary_"+model_name]=answer




################## ANSWER


answer=[]

for t,q in zip(df["text"],df["question"]):
    

    prompt_template = ChatPromptTemplate.from_template(template_string3)

    ##Lets plugin our constants from the template
    customer_messages = prompt_template.format_messages(question=q,text=t)

    
    #print(customer_messages)
    

    ##Let's get the chat response from the customer messages
    customer_response = chat.invoke(customer_messages)

    customer_response = re.sub(r"<think>.*?</think>\n?", "",customer_response, flags=re.DOTALL)

    print("ANNNNNNNNNNNNSSSSSSSSSSSSWWWWWWWWER QUESTION",customer_response)
    answer.append(customer_response)

df["answer_"+model_name]=answer





df.to_csv(filename)

