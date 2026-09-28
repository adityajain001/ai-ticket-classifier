import laya
from laya_test import questions

agent = laya.load("convaiinnovations/laya")

questions["ticket_category"]["criteria"]["OTHER"] = (
    "The message is unrelated to customer support, orders, accounts, "
    "payments, shipping, delivery, refunds, or subscriptions"
)

ood_messages = [
    "What's the weather like today?",
    "Who won the football match yesterday?",
    "Tell me a joke",
    "How do I make pasta?",
    "What is the capital of France?",
    "I am feeling bored today",
    "Explain Newton's second law",
    "Write a Python function to sort a list"
]

for message in (ood_messages):
    result = agent.predict(message,questions)
    prediction = result["answers"]["ticket_category"]["choice"]
    print(f"Message: {message}")
    print(f"Laya: {prediction}\n")