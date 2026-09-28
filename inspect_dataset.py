from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

# print(dataset)

train_data = dataset["train"]
train_data = train_data.class_encode_column("category")
split_dataset = train_data.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="category"
)

category_feature = split_dataset['train'].features["category"]

if __name__ == "__main__":
    # for i in [0, 1, 100]:
    #     print("Instruction:", train_data[i]["instruction"])
    #     print("Category:", train_data[i]["category"])
    #     print("Intent:", train_data[i]["intent"])
    #     print()

    #categories = set()
    def counts(train_data):
        category_counts = {}
        category_feature = train_data.features["category"]
        for row in train_data:
            temp = row["category"]
            temp = category_feature.int2str(temp)
            #categories.add(temp)
            if temp in category_counts:
                category_counts[temp]+=1
            else:
                category_counts[temp] = 1
        return category_counts
        

    # print(categories)
    # print("Number of categories:", len(categories))

    #print (category_counts)



    # print("TRAIN DATA")
    # train_counts = counts(split_dataset["train"])

    # for category, count in sorted(train_counts.items()):
    #     print(f"{category}: {count}")


    # print("\nTEST DATA")
    # test_counts = counts(split_dataset["test"])

    # for category, count in sorted(test_counts.items()):
    #     print(f"{category}: {count}")