from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import RagTokenizer, RagSequenceForGeneration, RagRetriever
from src.gen_vector_db import FAISSVectorDB, find_simposts_in_db
from torch import cuda
import torch
import requests
from src.secret import OPENAI_API_KEY
from openai import OpenAI


# Retrieving Similar posts and then prompting the LLM
# Please Note that this function does COT and RAG together
# There is a new method of combining RAG and COT - Retrieval Augmented Thoughts (RAT)
# https://medium.com/@bijit211987/rag-chain-of-thought-retrieval-augmented-thoughts-rat-3d3489517bf0

# In RAT a Zero Shot COT is the first step and the RAG based Context is used
# to improve the Step-by-Step process rather than just using RAG based Context
# to change the final answer.

# RAT also gives the opportunity to the LLM to decide what extent of RAG Context
# it should be inflluenced by.


openai_api = OpenAI(api_key = OPENAI_API_KEY)

def get_openai_response(query):
	draft_prompt = """
	NOTE: Answer the following question in a step-by-step manner.
	Be Careful! Answer the question in a structural manner and in several paragraphs. Use '\n\n' to
	split the answer into paragraphs. Respond to the question directly without any explanation or introductions anywhere  
	
	"""

	system_prompt = """You are GPT-3.5 Turbo. You will be answering a series of Questions and Answers like an 
						intelligent chatbot connected to a WordPress Blog
	"""

	zero_shot_COT= openai_api.chat.completions.create(
		model= "gpt-3.5-turbo",
		messages = [
			{
				"role" : "system",
				"content" : system_prompt
			},
			{
				"role" : "user",
				"content" : f"Question - {query}" + draft_prompt
			}
		],
		temperature= 0.3

		).choices[0].message.content 
	print("ZERO SHOT COT - ", zero_shot_COT)
	return zero_shot_COT

def gen_question(question, answer):
	quest_prompt = """
I want to verify the correctness of the given question, especially giving important to the last sentences.
Please summarize the content with a follow up question which will correctly correspond. You need to ensure that
some important keywords of the answer/content are included in this query. Give special importance to the last few sentences
**IMPORTANT**
Just output the query directly! Do not add explanations and introducement along with it. 
	"""

	system_prompt = """You are GPT-3.5 Turbo. You will be answering a series of Questions and Answers like an 
						intelligent chatbot connected to a WordPress Blog
	"""


	new_question = openai_api.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=[
			{
				"role": "system",
				"content": system_prompt
			},
			{
				"role": "user",
				"content": f"##Question: {question}\n\n##Content: {answer}\n\n##Instruction: {quest_prompt}"
			}
		],
		temperature = 1.0
	).choices[0].message.content

	print("New Query - ", new_question)
	return new_question

def revise_ans(question, answer, content):
	revise_dirn= """
I want to revise the answer according the content that has been provided. You need to keep in mind the following steps
If you find some errors in the answers, revise the answer to make it better according to the content
If you find some necessary details are ignored, add them to the answer.
If you the answer is correct and the content does not add anything significant, keep the original answer

IMPORTANT - Keep the structure as it is! (multiple paragrahps) in the revised answer.
Split the paragraphs only with '\n\n' characters. You need to make sure the answer is structural.
Output the revise answer directly without any explanations and introducements in your answer. 
	"""

	system_prompt = """You are GPT-3.5 Turbo. You will be answering a series of Questions and Answers like an 
						intelligent chatbot connected to a WordPress Blog
	"""

	revised = openai_api.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=[
				{
					"role": "system",
					"content": system_prompt
				},
				{
					"role": "user",
					"content": f"##Existing Text in Wiki Web: {content}\n\n##Question: {question}\n\n##Answer: {answer}\n\n##Instruction: {revise_dirn}"
				}
			],
		 temperature = 1.0
	).choices[0].message.content

	print("REVISION - ", revised)
	return revised

def generate_rag_response(db, query):

	#sim_posts = find_simposts_in_db(db, query, 10)
	
	zero_shot_COT = get_openai_response(query)
	steps= zero_shot_COT.split('\n\n')

	if(len(steps) == 0):
		return "Error in getting Initial COT steps"

	answer = ""
	
	for step_num, step_cont in enumerate(steps):
		answer = answer + '\n\n' + step_cont

		generated_query = gen_question(query, answer)

		similar_posts = find_simposts_in_db(db, generated_query, 3)
	
		for post in similar_posts:
			answer = revise_ans(query, answer, post)

	print("Final Answer - ", answer)
	return answer





if __name__ == "__main__":

	
	# for test purposes

	site_url = "https://legal-wires.com"
	embedding_model = "Alibaba-NLP/gte-large-en-v1.5"
	db = FAISSVectorDB(site_url, embedding_model, 256)
	#db.init_vector_db()

	#loads the db - for test purposes

	index= db.load_index_db()
	embedding_list = list(db.embeddings.values())
	similar_posts = find_simposts_in_db(db, "what is THE FATAL ACCIDENTS ACT, 1855", 5)

	query= "what is THE FATAL ACCIDENTS ACT, 1855"

	print(generate_rag_response(db,query,similar_posts))