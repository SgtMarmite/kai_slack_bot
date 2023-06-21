#TODO: Build out Zendesk Search Tooling

import streamlit as st
# Import the Zenpy Class
from zenpy import Zenpy
import datetime
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper
import os


creds = {
    'email' : st.secrets['ZENDESK_EMAIL'],
    'token' : st.secrets['ZENDESK_TOKEN'],
    'subdomain': st.secrets['ZENDESK_SUBDOMAIN']
}


# Default
zenpy_client = Zenpy(**creds)
# TODO: 
today = datetime.datetime.now()
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
pastMonth = datetime.datetime.now() - datetime.timedelta(days=30)
pastThreeMonths = datetime.datetime.now() - datetime.timedelta(days=90)
pastSixMonths = datetime.datetime.now() - datetime.timedelta(days=180)
pastYear = datetime.datetime.now() - datetime.timedelta(days=365)


def search_tickets_by_date_range(input_text: str, start_date: datetime, end_date: datetime):
    try:
        output = zenpy_client.search(input_text, created_between=[start_date, end_date], type='ticket')
    except Exception as e:
        print(f"Error while searching for tickets: {e}")
        return None

    for ticket in output:
        # Check that the ticket is not already in the data directory
        if not os.path.exists(f"data/{ticket.id}.txt"):
            # For each ticket that is returned, create a txt file with the ticket ID as the name in the data directory with the ticket's content
            try:
                with open(f"data/{ticket.id}.txt", "w") as f:
                    f.write("Ticket Number:\n" + str(ticket.id) + "\n--------------------\n\n")
                    f.write("Description:\n" + ticket.description + "\n--------------------\n\n")
                    f.write("Subject:\n" + ticket.subject + "\n--------------------\n\n")
                    f.write("Status:\n" + ticket.status + "\n--------------------\n\n")
                    comment_count = 0
                    for comment in zenpy_client.tickets.comments(ticket=ticket):
                        comment_count += 1
                        f.write("Comment " + str(comment_count) + ":\n")
                        f.write(comment.body + "\n--------------------\n\n")
            except Exception as e:
                print(f"Error while writing ticket {ticket.id} data: {e}")
                continue  # Continue with next ticket even if current ticket fails

    try:
        documents = SimpleDirectoryReader('data').load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
    except Exception as e:
        print(f"Error while creating index: {e}")
        return None

    return index

if __name__ == "__main__":
    search_tickets_by_date_range()