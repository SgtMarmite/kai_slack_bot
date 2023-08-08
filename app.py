import os
import logging

from decouple import config
from fastapi import FastAPI, Request
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.chat_engine.condense_question import ChatMessage
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.prompts import Prompt

from client.openai_client import OpenAIClient, AzureOpenAIClient
from client.base import AIClientException

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = openai.api_key = config("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = openai.api_base = config("OPENAI_API_BASE")


pinecone_api_key = config("PINECONE_API_KEY")
pinecone_env = config("PINECONE_ENV")
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

slack_bot_token = config("SLACK_BOT_TOKEN")
slack_bot_secret = config("SLACK_SIGNING_SECRET")
app_port = int(config("APP_PORT"))
data_dir = config("DATA_DIR")

custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")


class SlackApp:
    def __init__(self, bot_token, signing_secret):
        self.app = App(token=bot_token, signing_secret=signing_secret)
        self.app_handler = SlackRequestHandler(self.app)
        self.app.event("app_home_opened")(self.update_home_tab_wrapper)
        self.app.event("app_mention")(self.handle_message_events)
        self.app.event("message")(self.handle_message_events)

        vector_store = PineconeVectorStore(pinecone.Index("kaidev"))
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    @staticmethod
    def publish_home_tab(client, event, logger):
        try:
            client.views_publish(
                user_id=event["user"],
                view={
                    "type": "home",
                    "callback_id": "home_view",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Welcome to your _App's Home_* :tada:"
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "Write something to the bot and it will respond with the power of AI."
                            }
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error publishing home tab: {e}")

    def update_home_tab_wrapper(self, client, event, logger):
        self.publish_home_tab(client, event, logger)

    def handle_message_events(self, client, event):

        client.reactions_add(
            channel=event["channel"],
            name="eyes",
            timestamp=event["ts"]
        )
        message = event["text"]
        logging.info(f"Message received: {message}")

        thread_ts = event.get("thread_ts", event["ts"])

        thread_history = client.conversations_replies(
            channel=event["channel"],
            ts=thread_ts
        )

        messages = thread_history["messages"]
        bot_messages = []
        user_messages = []

        for msg in messages:
            if msg.get("bot_id", False):
                bot_messages.append(msg.get("text", ""))
            else:
                user_messages.append(msg.get("text", ""))

        message_history = []

        for user_message, bot_message in zip(user_messages[:len(user_messages)-1], bot_messages):
            user_message_object = ChatMessage(role="user", content=user_message)
            bot_message_object = ChatMessage(role="assistant", content=bot_message)
            message_history.append(user_message_object)
            message_history.append(bot_message_object)

        print(f"Message history: {message_history}")

        query_engine = self.index.as_query_engine()
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=message_history,
            verbose=True
        )

        response = chat_engine.chat(message)

        response_str = str(response)
        response_source = response.sources

        response_message = f"Response: {response_str}\n\nSources: {response_source}"

        logging.info(f"Returning response: {response_str}")

        client.chat_postMessage(
            channel=event["channel"],
            thread_ts=event["ts"],
            text=response_message
        )


api = FastAPI()
slack_app = SlackApp(slack_bot_token, slack_bot_secret)


@api.post("/slack/events")
async def endpoint(req: Request):
    return await slack_app.app_handler.handle(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:api", host="0.0.0.0", port=app_port)
