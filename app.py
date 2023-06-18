import os
# Use the package we installed
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.tools.confluence_search.confluence_search import conflu_search
from decouple import config


slack_bot_token = config("SLACK_BOT_TOKEN")
slack_bot_secret = config("SLACK_SIGNING_SECRET")
app_port = int(config("APP_PORT"))
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Initializes your app with your bot token and signing secret
app = App(
    token=slack_bot_token,
    signing_secret=slack_bot_secret
)


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
        # views.publish is the method that your app uses to push a view to the Home tab
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view={
                "type": "home",
                "callback_id": "home_view",

                # body of the view
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
                            "text": "Write something to the bot and it will respond."
                        }
                    }
                ]
            }
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("home_button_click")
def handle_button_click(ack, body, client):
    try:
        # Acknowledge the button click
        ack()

        # Get the user ID and channel ID
        user_id = body["user"]["id"]
        channel_id = body["container"]["channel_id"]

        # Get the message text from the button click event
        message_text = body["message"]["text"]

        # Respond back to the user with the message they wrote
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=body["container"]["message_ts"],
            text=f"Hi <@{user_id}> :wave:\nYou wrote: {message_text}"
        )
    except Exception as e:
        print(f"Error handling button click: {e}")


@app.event("app_mention")
def answer_question(client, event):
    # Add a reaction to the message
    client.reactions_add(
        channel=event["channel"],
        name="eyes",
        timestamp=event["ts"]
    )

    # Get the message text from the event
    message = event["text"]
    print(f"Message received: {message}")

    # Perform Confluence search and get the response
    response = conflu_search(message).as_query_engine().query(message)

    print(f"Returning response: {response}")

    # Respond back to the user with the search results
    client.chat_postMessage(
        channel=event["channel"],
        thread_ts=event["ts"],
        text=f"Hi <@{event['user']}> :wave:\n{response}"
    )


@app.event("message")
def handle_message_events(client, event):
    # Add a reaction to the message
    client.reactions_add(
        channel=event["channel"],
        name="eyes",
        timestamp=event["ts"]
    )

    # Get the message text from the event
    message = event["text"]
    print(f"Message received: {message}")

    # Perform Confluence search and get the response
    response = conflu_search(message).as_query_engine().query(message)

    print(f"Returning response: {response}")

    # Respond back to the user with the search results
    client.chat_postMessage(
        channel=event["channel"],
        thread_ts=event["ts"],
        text=f"Hi <@{event['user']}> :wave:\n{response}"
    )


# Start your app
if __name__ == "__main__":
    app.start(app_port)
