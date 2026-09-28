import laya

agent = laya.load("convaiinnovations/laya")

custom_messages = [
    "Amazon charged me but the checkout page says the payment failed",
    "Been waiting two weeks and the parcel still isn't here",
    "Where can I get a copy of what I was billed for?",
    "I moved houses yesterday, where do I update where you send my stuff?",
    "Take my card off this account",
    "I want my money back",
    "Your product is terrible",
    "I can't get into my profile"
]

from laya_test import questions


for message in (custom_messages):
    result = agent.predict(message,questions)
    prediction = result["answers"]["ticket_category"]["choice"]
    print(f"Message: {message}")
    print(f"Laya: {prediction}\n")