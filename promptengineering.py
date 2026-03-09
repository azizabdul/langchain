import os
from constant import openai_key

os.environ["OPENAI_API_KEY"] = openai_key

from langchain_core.prompts import PromptTemplate

demo_template='''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt=PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
    )

prompt.format(financial_concept='income tax')