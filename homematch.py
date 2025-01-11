import os

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

from langchain.llms import OpenAI