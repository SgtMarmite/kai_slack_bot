from atlassian import Confluence, errors
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from langchain.llms import OpenAI
import os
import sys
from decouple import config
from typing import Union
import logging

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


confluence = Confluence(
    url=config("CONFLUENCE_URL"),
    username=config("CONFLUENCE_USERNAME"),
    password=config("CONFLUENCE_PASSWORD")
)

main_file_path = sys.modules['__main__'].__file__
main_dir = os.path.dirname(os.path.abspath(main_file_path))


pre_prompt = """
You are being given a question that is meant to be answered by searching the 
Keboola Confluence docs. The only way to answer this question is to search the docs. 
In order to search the docs, you must generate a CQL query that will 
return the most relevant results. 
Confluence uses Apache Lucene for text indexing, 
which provides a rich query language. 
Much of the information about how to generate a CQL query 
is derived from the Query Parser Syntax
page of the Lucene documentation.


A query is broken up into terms and operators. 
There are two types of terms: Single Terms and Phrases.

A Single Term is a single word such as "test" or "hello".

A Phrase is a group of words surrounded by double quotes such as "hello dolly".

Note: All query terms in Confluence are case insensitive.

Your task is to take the input question and generate a CQL query that will return the most relevant results. 
Use shorter terms rather than long ones.
Respond only with a valid atlassian CQL query!

Examples:
text ~ "Keboola AND Python AND component"
text ~ "BYODB AND process"
"""


def generate_cql_query_keywords(input_text: str) -> str:
    llm = OpenAI()
    response = llm.predict(pre_prompt + input_text)
    cql_query = response.replace("\n", "").strip(" ")
    return cql_query


def query_conflu(cql_query: str):
    print(f"Query: {cql_query}")

    pages = None
    try:
        pages = confluence.cql(cql_query)
    except errors.ApiValueError:
        logging.error(f"Query: {cql_query} is invalid.")

    return pages


def download_documents(pages: list):
    if pages:
        print(f"Found pages: {pages}")
        documents = []

        for page in pages:
            print(f"Checking page: {page})")
            # Check the local directory to see if we already have the page's content
            if os.path.exists(f"{main_dir}/data/{page['content']['id']}.txt"):
                with open(f"{main_dir}/data/{page['content']['id']}.txt", "r") as f:
                    documents.append(f.read())
                    f.close()
                continue

            # If we don't have the page's content, then get it from Confluence
            else:
                content = confluence.get_page_by_id(page['content']['id'], expand='body.view')
                documents.append(content['body']['view']['value'])
                # add each page's content as a txt file in the data directory
                with open(f"{main_dir}/data/{page['content']['id']}.txt", "w") as f:
                    f.write(content['body']['view']['value'])
                    f.close()
        # convert documents to a string
        print(f"Using document directory: {main_dir}/data")
        documents = SimpleDirectoryReader(f"{main_dir}/data").load_data()
        index = GPTVectorStoreIndex.from_documents(documents)

        return index
    else:
        return None


def conflu_search(search: str) -> Union[GPTVectorStoreIndex, None]:
    while True:
        query = generate_cql_query_keywords(search)
        r = query_conflu(query)
        if r is not None and r.get("results"):
            break
    index = download_documents(r.get("results"))
    return index


if __name__ == "__main__":
    conflu_search("What is the complete BYODB process?")


