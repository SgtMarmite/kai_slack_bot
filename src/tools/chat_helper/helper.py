import openai
import json
from typing import Union

default_content = "You are a helpful Slack assistant used to help navigate our company knowledgebase. You were created"\
                  "by Poseidon himself to help Keboola employees to help all employees answer their questions."


def gpt_query(prompt: str, user_history: list = None, bot_history: list = None) -> Union[str, None]:

    default_messages = [
                {"role": "system", "content": default_content},
                {"role": "user", "content": prompt}
            ]
    messages_history = None

    if user_history and bot_history:
        messages_history = [{"role": "system", "content": default_content}]
        for user, bot in zip(user_history, bot_history):
            messages_history.append({"role": "user", "content": user})
            messages_history.append({"role": "system", "content": bot})
        messages_history.append({"role": "user", "content": prompt})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=default_messages if not messages_history else messages_history,
            max_tokens=256,
            temperature=0
        )
    except openai.error.ServiceUnavailableError as e:
        return str(e)

    response = str(response["choices"][0]["message"])
    c = json.loads(response)

    if c.get("content", None):
        return str(c.get("content"))

    return None


def is_kb_question(text: str) -> bool:
    prompt = f"Classify the following text if it is a knowledgebase question or anything else. Respond only with " \
             f"knowledgebase/other:\n\n{text}\n\nLabel:"
    result = gpt_query(prompt)
    return result == "knowledgebase" if result else False


if __name__ == "__main__":
    # Example usage
    text_input = "How can I create a Python component in Keboola?"
    is_kb = is_kb_question(text_input)
    print(is_kb)
