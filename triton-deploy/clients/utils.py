def add_system():
    return "You are a smart chatbot (Bot) of humans (User).\n"+ \
    "your answers need to useful and harmless.\n\n"

def build_prompt(content):
    if len(content) < 50:
        prompt = f"<s>{add_system()}User: {content}\n\nBot: "
    else:
        prompt = f"<s>User: {content }\n\nBot: "
    return prompt
def str_as_bytes(text:str):
    return text.encode("utf-8")