# Responsible AI Report

## 1. Executive Summary

This project develops a Google Play Store sentiment analysis system that
classifies user reviews into three sentiment categories: Positive,
Negative, and Neutral.

The system uses TF-IDF for text representation and a Linear Support
Vector Machine (SVM) for classification. LIME and SHAP were used as
explainability techniques to understand model behavior.

Responsible AI considerations were evaluated across fairness, privacy,
consent, transparency, explainability, performance, robustness,
security, human oversight, and monitoring.

The purpose of this report is to document the responsible use,
limitations, and potential risks of the developed sentiment analysis
system.

---

## 2. System Description

### Application

The application accepts a Google Play Store review as input and predicts
its sentiment.

### Processing Pipeline

The system follows this pipeline:

1. User enters a review.
2. Text preprocessing is applied.
3. TF-IDF converts the text into numerical features.
4. The trained Linear SVM model predicts the sentiment.
5. LIME can provide a local explanation for an individual prediction.
6. SHAP was used during the explainability experiment to study feature
   importance.

### Sentiment Classes

The model supports:

- Positive
- Negative
- Neutral

---

## 3. Dataset and Data Considerations

The experiment used a Google Play Store reviews dataset containing
31,465 reviews.

The available dataset contains fields including:

- app_id
- review_id
- user_name
- review_text
- review_score
- thumbs_up_count
- review_date

The review text was used as the main input for sentiment analysis.

The dataset contains user-related fields such as `user_name`.
Therefore, these fields should not be unnecessarily exposed or used as
model features when they are not required for sentiment prediction.

The project does not use explicit demographic attributes such as age,
gender, religion, ethnicity, or income for prediction.

---

## 4. Model and Methodology

The model uses:

- TF-IDF Vectorization
- Unigrams and Bigrams
- Maximum 30,000 features
- Linear SVM classifier
- Class weighting to handle class imbalance
- Stratified train-test split
- 20% test data
- Random state = 42

The text preprocessing includes lowercasing, URL removal, HTML tag
removal, removal of unwanted characters, and whitespace normalization.

---

## 5. Model Performance

The model was evaluated on the test dataset.

The reported results from the experiment were:

| Metric | Result |
|---|---:|
| Accuracy | 85.62% |
| Weighted F1-Score | 84.88% |
| Macro F1-Score | 60.76% |

### Interpretation

The accuracy indicates that approximately 85.62% of the test predictions
were correct.

The Weighted F1-score accounts for the relative number of samples in
each class.

The Macro F1-score gives equal importance to each sentiment class.
The difference between the Weighted F1-score and Macro F1-score indicates
that performance was not uniform across the sentiment classes.

The Neutral class showed substantially lower performance than the other
sentiment classes during the experiment.

Therefore, accuracy alone should not be used to determine model
reliability.

---

## 6. Fairness Assessment

Fairness is important because machine learning models can behave
differently for different groups or types of users.

The available dataset does not contain explicit sensitive demographic
attributes. Therefore, demographic fairness metrics could not be
calculated from the available experiment.

Potential sources of uneven performance include:

- Different writing styles
- Short reviews
- Informal language
- Spelling variations
- Sarcasm
- Mixed-language reviews
- Different types of applications

The model should therefore not be assumed to perform equally for every
possible user or review type.

### Fairness Mitigation

Future evaluation can include:

- Testing performance across different language groups when appropriate
  data is available.
- Comparing precision, recall, and F1-score across relevant subsets.
- Checking whether certain review categories consistently receive more
  incorrect predictions.
- Expanding the training dataset with more diverse review examples.

---

## 7. Privacy Assessment

The application accepts review text from users.

Users should avoid entering:

- Personal identifiers
- Phone numbers
- Email addresses
- Passwords
- Financial information
- Confidential information

The application should avoid unnecessary storage or logging of raw
user-provided reviews.

If review data is stored for future analysis or model training, the data
should be handled according to the applicable privacy requirements.

---

## 8. Consent and Data Usage

The current application uses entered review text to generate a
sentiment prediction.

If additional user data is collected, stored, or reused, users should
be informed about:

- What data is collected
- Why the data is collected
- How the data will be used
- Whether the data will be stored
- Whether the data will be used for future model training

Data should not be reused for unrelated purposes without appropriate
permission or consent.

---

## 9. Transparency

The system provides a clear machine learning pipeline:

**Review → Text Cleaning → TF-IDF → Linear SVM → Sentiment**

The model type, feature extraction method, and sentiment classes are
known and documented.

However, machine learning predictions are not guaranteed to be correct.

Users should understand that the sentiment returned by the application
is a prediction generated from patterns learned from the training data.

---

## 10. Explainability

Explainability techniques were included to improve understanding of
model predictions.

### LIME

LIME provides local explanations for individual predictions.

It identifies words or features that contributed to the prediction of a
specific review.

This helps users understand why a particular review was classified as
Positive, Negative, or Neutral.

### SHAP

SHAP was used during the model explainability experiment to analyze
feature importance.

SHAP helps identify features that have a stronger influence on model
behavior.

Together, LIME and SHAP provide additional transparency into the
otherwise difficult-to-interpret machine learning model.

### Limitation of Explainability

LIME and SHAP explanations should not be interpreted as a perfect
representation of the internal reasoning of the model. They are
explanation methods that approximate or describe model behavior.

---

## 11. Robustness and Limitations

The model may have difficulty with certain types of reviews.

Potential limitations include:

### Sarcasm

A review such as "Amazing, another update that broke everything"
contains language that may be difficult for a sentiment classifier to
interpret correctly.

### Very Short Reviews

Reviews containing only a few words may not provide enough information
for reliable classification.

### Informal Language

Slang, abbreviations, spelling errors, and unusual expressions may
affect prediction quality.

### Mixed Sentiment

A review may contain both positive and negative statements, making
classification difficult.

### Language Variation

The model may not perform equally well on languages or writing styles
that were insufficiently represented in the training data.

### Domain Variation

The model was developed using Google Play Store reviews. Its performance
may differ when applied to reviews from other domains.

---

## 12. Security Considerations

The application should safely handle user-provided input.

Security considerations include:

- Validate user input.
- Avoid unnecessary logging of raw reviews.
- Avoid exposing sensitive information.
- Keep application dependencies updated.
- Protect the trained model and application files from unauthorized
  modification.

---

## 13. Human Oversight

The system provides automated sentiment predictions.

The prediction should not be treated as an unquestionable decision.

Users should consider the original review and its context when
interpreting the prediction.

Human review is particularly important when the prediction may influence
an important decision.

The system is intended primarily for sentiment analysis,
experimentation, and educational purposes.

---

## 14. Monitoring and Drift

Machine learning models can experience changes in input data over time.

For example, Google Play Store reviews may change because of:

- New applications
- New slang
- Changes in user behavior
- Changes in review patterns
- Changes in language usage

The project includes monitoring considerations for input data.

A future production implementation should monitor:

- Distribution of predicted sentiment
- Review length
- Vocabulary changes
- Prediction confidence or decision scores
- Model performance when labelled data becomes available

A formal statistical drift analysis was not performed as part of this
experiment. Therefore, future versions can incorporate statistical
methods such as distribution-based drift tests.

---

## 15. Responsible Use

The system should be used for:

- Sentiment analysis
- Educational demonstrations
- Software experimentation
- Exploratory analysis of user reviews

The system should not be used as the sole basis for:

- High-impact decisions about individuals
- Profiling users
- Determining a person's character
- Making decisions about employment, finance, healthcare, or other
  sensitive areas

---

## 16. Risk-Mitigation Checklist

| Responsible AI Area | Status | Mitigation / Consideration |
|---|---|---|
| Fairness | ⚠️ Requires monitoring | No demographic attributes available for fairness testing |
| Privacy | ✅ Addressed | Avoid collecting unnecessary personal information |
| Consent | ✅ Addressed | Inform users about data collection and reuse |
| Transparency | ✅ Addressed | Model and processing pipeline documented |
| Explainability | ✅ Implemented | LIME and SHAP used |
| Performance | ✅ Evaluated | Accuracy and F1 metrics reported |
| Robustness | ⚠️ Limitation identified | Sarcasm, short text and language variation can affect results |
| Security | ⚠️ Requires maintenance | Validate inputs and update dependencies |
| Human Oversight | ✅ Addressed | Predictions should not replace human judgment |
| Monitoring | ⚠️ Requires future work | Monitor input and prediction distributions |

---

## 17. Future Improvements

The following improvements can strengthen the responsible use of the
system:

1. Perform fairness analysis on appropriate demographic or linguistic
   groups when ethically and legally suitable data is available.
2. Evaluate the model on additional datasets.
3. Add formal statistical drift detection.
4. Monitor model performance after deployment.
5. Improve handling of sarcasm and mixed sentiment.
6. Support multilingual reviews where sufficient training data is
   available.
7. Periodically retrain and evaluate the model using representative
   data.
8. Minimize storage of user-provided review text.

---

## 18. Conclusion

The Google Play Store sentiment analysis system incorporates several
Responsible AI considerations including transparency, privacy,
explainability, performance evaluation, human oversight, and recognition
of model limitations.

The use of LIME and SHAP improves the interpretability of the model,
while the documented performance metrics provide evidence of model
behavior.

However, the system has limitations, particularly in class-wise
performance, language variation, sarcasm, and generalization.

Responsible deployment therefore requires continuous evaluation,
appropriate monitoring, privacy protection, and human oversight.