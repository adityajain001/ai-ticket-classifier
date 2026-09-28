# AI Customer Support Ticket Classifier

An end-to-end machine learning system that classifies customer-support messages into 11 categories using semantic sentence embeddings and serves predictions through a FastAPI REST API packaged with Docker.

## Problem

Customer-support systems receive large volumes of messages covering issues such as payments, deliveries, refunds, account access, and shipping. Manually routing these tickets is time-consuming.

This project automatically classifies a customer message into one of 11 support categories:

- ACCOUNT
- CANCEL
- CONTACT
- DELIVERY
- FEEDBACK
- INVOICE
- ORDER
- PAYMENT
- REFUND
- SHIPPING
- SUBSCRIPTION

## Approach

I compared two text-classification approaches:

### 1. Bag-of-Words Baseline

Customer messages are converted into binary Bag-of-Words features using `CountVectorizer`, followed by a Logistic Regression classifier.

### 2. Sentence Embeddings

Customer messages are encoded into 384-dimensional semantic embeddings using the pretrained `all-MiniLM-L6-v2` Sentence Transformer. A Logistic Regression classifier is then trained on these embeddings.

The final inference pipeline is:

Customer Message → MiniLM Embedding → Logistic Regression → Support Category

## Results

The dataset contains 26,872 customer-support messages. I used a stratified 80/20 train-test split, resulting in 21,497 training examples and 5,375 test examples.

| Model | Feature Dimension | Test Accuracy | Test Errors |
|---|---:|---:|---:|
| Binary Bag-of-Words + Logistic Regression | 2,486 | 99.67% | 18 |
| MiniLM Embeddings + Logistic Regression | 384 | 99.80% | 11 |

MiniLM reduced the number of test-set errors from 18 to 11, corresponding to a 38.9% reduction in errors.

### Robustness Testing

Because the dataset is highly structured and produced unusually high test accuracy, I also tested both models on eight manually written messages using more natural and indirect phrasing.

Examples included:

- `"Been waiting two weeks and the parcel still isn't here"`
- `"Where can I get a copy of what I was billed for?"`
- `"Your product is terrible"`

On this small qualitative test set:

- Bag-of-Words correctly classified 4/8 messages.
- MiniLM correctly classified 8/8 messages.

This hand-written test is intended as a qualitative robustness check, not a statistically representative benchmark. It showed that semantic embeddings handled indirect phrasing better than the word-based baseline.

## Zero-Shot Classification Experiment

I also evaluated a local Laya model as a zero-shot classifier. Unlike the supervised
BoW and MiniLM classifiers, Laya was not trained on the 21,497 training tickets.
Instead, it was given natural-language descriptions of the 11 support categories
and asked to choose the most appropriate category.

On the same 5,375-message test set:

| Approach | Task-specific training examples | Accuracy |
|---|---:|---:|
| Bag-of-Words + Logistic Regression | 21,497 | 99.67% |
| MiniLM + Logistic Regression | 21,497 | 99.80% |
| Laya (zero-shot) | 0 | 85.28% |

Laya correctly classified 4,584 of 5,375 test messages.

### Error Analysis

The zero-shot model's largest confusions included:

- `ORDER → CANCEL`: 193 examples
- `DELIVERY → SHIPPING`: 190 examples

Manual inspection showed that some apparent errors reflected differences between
the dataset's labeling conventions and the natural-language category definitions.

For example, messages such as:

> "i want help to cancel order {{Order Number}}"

were labeled `ORDER` in the dataset, while the zero-shot model selected `CANCEL`.

Similarly, some messages containing phrases such as "shipping period" were labeled
`DELIVERY` by the dataset while Laya selected `SHIPPING`.

This illustrates an important distinction between the two approaches: a supervised
classifier can learn dataset-specific label conventions from examples, while a
zero-shot classifier must infer the intended taxonomy from the supplied category
descriptions.

### Robustness and Out-of-Distribution Tests

On the same eight manually written in-domain messages used for qualitative
robustness testing:

| Model | Correct |
|---|---:|
| Bag-of-Words + Logistic Regression | 4/8 |
| Laya (zero-shot) | 6/8 |
| MiniLM + Logistic Regression | 8/8 |

These eight examples are a qualitative robustness check, not a statistical
estimate of real-world accuracy.

I also added an `OTHER` category described in natural language and tested Laya on
eight clearly out-of-domain messages covering topics such as weather, sports,
cooking, general knowledge, and programming. Laya assigned all 8/8 examples to
`OTHER`. This is also a small qualitative test and should not be interpreted as
100% out-of-distribution detection accuracy.

## REST API

The trained classifier is exposed through a FastAPI REST API.

### Prediction Endpoint

`POST /predict`

Example response:

```json
{
  "category": "DELIVERY"
}
```

Input validation is performed using Pydantic. Empty or whitespace-only messages are rejected before model inference.

FastAPI also provides interactive API documentation at `/docs`.

## Docker

The application can be packaged and run using Docker.

Build the image:

```bash
docker build -t ai-ticket-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 ai-ticket-classifier
```

The API documentation is then available at `http://localhost:8000/docs`.

## Project Structure

```text
ai-ticket-classifier/
├── api.py                       # FastAPI application
├── predict.py                   # Inference pipeline
├── embedding_model.py           # MiniLM classifier training
├── baseline_model.py            # Bag-of-Words baseline
├── compare_models.py            # Custom robustness comparison
├── inspect_dataset.py           # Dataset preparation and splitting
├── embedding_classifier.joblib  # Saved Logistic Regression classifier
├── requirements.txt             # Runtime dependencies
├── Dockerfile                   # Container configuration
└── README.md
```

## Limitations

- The training dataset is highly structured, so the very high held-out accuracy may not represent performance on real-world customer messages.
- The classifier is closed-set: every valid input must be assigned to one of the 11 categories.
- Out-of-domain messages such as `"whats up"` can therefore receive an unrelated category instead of being rejected.
- The eight-message robustness set is useful for qualitative testing but is too small to estimate real-world accuracy.
- The current system does not implement confidence-based abstention or an `UNKNOWN` / `OTHER` category.

## Future Work

- Add confidence scores and investigate a validated abstention threshold.
- Add an `UNKNOWN` or out-of-domain detection mechanism.
- Build a larger robustness test set containing paraphrases, typos, and out-of-domain messages.
- Compare the current pipeline with additional decision/classification models.
- Optimize the Docker image for CPU inference.