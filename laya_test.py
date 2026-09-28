import laya
# state = {
#     "message": "My parcel still hasn't arrived"
#}

questions = {
    "ticket_category": {
        "type": "choice",
        "instructions": "Which category best describes the customer's primary request or problem?",
        "criteria": {
            "ACCOUNT": "Account access, signup, login, profile, or account management issues",
            "CANCEL": "Requests to cancel an order, service, or transaction",
            "CONTACT": "Requests for contact information or ways to reach support",
            "DELIVERY": "Problems about delivery status, arrival, or delivery time",
            "FEEDBACK": "Customer feedback, complaints, praise, or opinions",
            "INVOICE": "Questions about invoices, bills, or invoice documents",
            "ORDER": "Questions about placing, managing, or purchasing an order",
            "PAYMENT": "Problems involving payments, charges, checkout, or payment methods",
            "REFUND": "Requests about refunds, reimbursements, or getting money back",
            "SHIPPING": "Questions about shipping methods, addresses, or shipment arrangements",
            "SUBSCRIPTION": "Questions or requests concerning subscriptions",
            #"OTHER": "Not a customer support request covered by the other categories"
        }
    }
}

#result = agent.predict(state, questions)

#print(result)
def run_benchmark():
    
    from inspect_dataset import split_dataset
    import time

    agent = laya.load("convaiinnovations/laya")

    test_messages = split_dataset["test"]["instruction"]
    test_labels = split_dataset["test"]["category"]

    category_feature = split_dataset["test"].features["category"]

    correct = 0
    laya_predictions = []

    import json

    start_time = time.time()

    for i in range(len(test_messages)):
        state = {"message": test_messages[i]}

        result = agent.predict(state, questions)
        prediction = result["answers"]["ticket_category"]["choice"]

        actual = category_feature.int2str(test_labels[i])
        laya_predictions.append(prediction)

        if prediction == actual:
            correct += 1

        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(test_messages))
            with open("laya_predictions.json", "w") as f:
                json.dump(laya_predictions, f)

    with open("laya_predictions.json", "w") as f:
        json.dump(laya_predictions, f)

    end_time = time.time()

    print("Correct:", correct, "/", len(test_messages))

    print("Accuracy:", correct / len(test_messages))

    print("Time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    run_benchmark()
