# Part A: IMDb Movie Review Sentiment Analysis

## Video Presentation: [Click here](https://drive.google.com/file/d/1_Yztv1Q1yPcr7Qz_1SsepJ4hnrbm4fOY/view?usp=drive_link)


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import uniform
```


```python
# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     C:\Users\abelj\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\abelj\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\abelj\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
imdb_data = pd.read_csv("data_imdb.csv")
imdb_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Data Exploration and Preprocessing

### Analyze the dataset for trends, missing values, and outliers

#### a. Missing values


```python
imdb_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   review     50000 non-null  object
     1   sentiment  50000 non-null  object
    dtypes: object(2)
    memory usage: 781.4+ KB
    

- There are 50,000 rows and no missing values.

#### b. Unique reviews


```python
imdb_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>49581</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Loved today's show!!! It was a variety and not...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5</td>
      <td>25000</td>
    </tr>
  </tbody>
</table>
</div>



- 419 reviews are not unique and need to be removed.

#### c. Sentiment distribution


```python
imdb_data['sentiment'].value_counts()
```




    sentiment
    positive    25000
    negative    25000
    Name: count, dtype: int64



- The sentiment classes are balanced with each having 25,000 records.

#### d. Review length outliers


```python
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

sns.boxplot(imdb_data['review'].apply(len), ax=axes[0])
axes[0].set_title("Review Length Outliers")

sns.histplot(imdb_data['review'].apply(len), bins=50, kde=True, ax=axes[1])
axes[1].set_title("Review Length Distribution")

plt.tight_layout()
plt.show()
```


    
![png](README_Resources/output_17_0.png)
    


- There are so many outliers too far away from the upper limit.
- The review length is positively skewed.

- The review length is positively skewed.

### Perform data cleaning and text preprocessing

#### a. Keep only unique reviews


```python
imdb_data_unique = imdb_data.drop_duplicates(subset='review', keep='first')
imdb_data_unique.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>49581</td>
      <td>49581</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>49581</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>24884</td>
    </tr>
  </tbody>
</table>
</div>



- The first occurrence was kept and other non-unique reviews were dropped.

#### b. Remove review length outliers


```python
# Calculate Q1 and Q3
Q1 = imdb_data_unique['review'].apply(len).quantile(0.25)
Q3 = imdb_data_unique['review'].apply(len).quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
imdb_data_filtered = imdb_data_unique[(imdb_data_unique['review'].apply(len) >= lower_bound) & 
                                 (imdb_data_unique['review'].apply(len) <= upper_bound)
]
```


```python
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

sns.boxplot(imdb_data_filtered['review'].apply(len), ax=axes[0])
axes[0].set_title("Review Length Outliers After Filtering")

sns.histplot(imdb_data_filtered['review'].apply(len), bins=50, kde=True, ax=axes[1])
axes[1].set_title("Review Length Distribution After Filtering")

plt.tight_layout()
plt.show()
```


    
![png](README_Resources/output_26_0.png)
    


- After removing outliers, reviews with extreme high lengths have been removed.
- Review length is less positively skewed.


```python
imdb_data_filtered.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45876</td>
      <td>45876</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>45876</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>23029</td>
    </tr>
  </tbody>
</table>
</div>



- After removing outliers, 4124 records have been removed.

#### c. Reviews with extremely low review lengths


```python
imdb_data_filtered[(imdb_data_filtered['review'].apply(len) < 30)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26486</th>
      <td>#ERROR!</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>



- The #ERROR! review can be removed as it is an error code rather than a review.


```python
# Bad reviews removed
imdb_data_filtered = imdb_data_filtered[(imdb_data_filtered['review'].apply(len) > 30)]
```

#### d. Lowercase the text


```python
imdb_data_filtered['processed_review'] = imdb_data_filtered['review'].apply(
    lambda x: x.lower()
)
```

#### e. Remove HTML tags


```python
imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: re.sub('<.*?>', '', x)
)
```

#### f. Remove punctutations and special characters


```python
imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: re.sub(r'[^\w\s]', '', x)
)
```

#### g. Tokenize the test


```python
imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: word_tokenize(x)
)
```

#### h. Remove stop words


```python
stop_words = set(stopwords.words('english'))

imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: [word for word in x if word not in stop_words]
)
```

#### i. Lemmatize the tokens


```python
lemmatizer = WordNetLemmatizer()

imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)
```

#### j. Remove numbers and words with 2 or less characters


```python
# Remove the numbers and words, 2 characters or less
imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: [re.sub(r'\b\w{1,2}\b|\d+', '', word) for word in x]
)

# Get rid of empty strings
imdb_data_filtered['processed_review'] = imdb_data_filtered['processed_review'].apply(
    lambda x: [word for word in x if word != '']
)
```

#### k. Word cloud representation for positive and negative reviews


```python
# Word frequency analysis
positive_reviews = imdb_data_filtered[imdb_data_filtered['sentiment'] == 'positive']['processed_review']
negative_reviews = imdb_data_filtered[imdb_data_filtered['sentiment'] == 'negative']['processed_review']

# Join all lists of words into a single string for positive reviews
text_pos = ' '.join([word for review in positive_reviews for word in review])

# Join all lists of words into a single string for negative reviews
text_neg = ' '.join([word for review in negative_reviews for word in review])

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Create a word cloud object for positive reviews
wordcloud_pos = WordCloud(width=800, height=400).generate(text_pos)

# Create a word cloud object for negative reviews
wordcloud_neg = WordCloud(width=800, height=400).generate(text_neg)

# Plot the word cloud for positive reviews on the first subplot
axs[0].imshow(wordcloud_pos, interpolation='bilinear')
axs[0].set_title('Positive Reviews')
axs[0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[1].imshow(wordcloud_neg, interpolation='bilinear')
axs[1].set_title('Negative Reviews')
axs[1].axis('off')

# Layout so plots do not overlap
fig.tight_layout()

plt.show()
```


    
![png](README_Resources/output_49_0.png)
    


- Both positive and negative reviews have words like "movie", "film", "story", "one", etc.
- For positive reviews, words "love", "work", "life" is visible while they are not visible in the negative reviews.
- For negative reviews, words like "scene", "plot", "actor" is visible while they are not visible in positive reviews.

## 2. Feature Engineering

### a. Transform textual data into numbers for model use using TF-IDF


```python
# Join words for vectorization
imdb_data_filtered['processed_text'] = imdb_data_filtered['processed_review'].apply(
    lambda x: ' '.join(x)
)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(imdb_data_filtered['processed_text'])
```

- It is not necessary to engineer features like word count, character count or average word length as TF-IDF captures a lot of rich information from text data.

## 3. Model Development and Evaluation

### a. Encode the categorical variable


```python
for column in ['sentiment']:
    le = LabelEncoder()
    imdb_data_filtered[column] = le.fit_transform(imdb_data_filtered[column])
```

### b. Split data to train and test sets


```python
# Prepare data
X = tfidf_features
y = imdb_data_filtered['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### c. Train Random Forest Classifier


```python
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
```

#### c.1. Evaluate Random Forest Classifier


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

    Accuracy: 0.848283378746594
    Precision: 0.8420707732634338
    Recall: 0.8521220159151194
    F1 Score: 0.8470665787738958
    ROC-AUC: 0.8483357875748463
    


    
![png](README_Resources/output_63_1.png)
    



    
![png](README_Resources/output_63_2.png)
    


### d. Train Support Vector Classifier


```python
# Create and train model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)
```

#### d.1. Evaluate Support Vector Classifier


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("SVC Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

    Accuracy: 0.8934059945504087
    Precision: 0.8780383795309168
    Recall: 0.9102564102564102
    F1 Score: 0.8938571738658563
    ROC-AUC: 0.8936360529028772
    


    
![png](README_Resources/output_67_1.png)
    



    
![png](README_Resources/output_67_2.png)
    


### e. Train Logistic Regression


```python
# Train logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
# Predict labels on the test set
y_pred = lr.predict(X_test)
```

#### e.1. Evaluate Logistic Regression


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

    Accuracy: 0.8900272479564033
    Precision: 0.8746535919846514
    Recall: 0.9069407603890363
    F1 Score: 0.8905046120455778
    ROC-AUC: 0.8902581677670832
    


    
![png](README_Resources/output_71_1.png)
    



    
![png](README_Resources/output_71_2.png)
    


### f. Train Naive Bayes Classifier


```python
# Train the Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)
```

#### f.1. Evaluate Naive Bayes


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

    Accuracy: 0.8638692098092643
    Precision: 0.879138689511461
    Recall: 0.8393015030946065
    F1 Score: 0.858758339929888
    ROC-AUC: 0.86353378745356
    


    
![png](README_Resources/output_75_1.png)
    



    
![png](README_Resources/output_75_2.png)
    


## 4. Model Evaluation

- Support Vector Machine is the most optimal model with f1-score 0.893.
- ROC-AUC of 0.893 means the model has high discriminatory power.
- False positives: 572, False negatives: 406, which is slightly better that logistic regression.
- Of all reviews predicted as positive, 87.8% were actually positive.
- Of all actual positive reviews, 91% were correctly identified.

# Part B: News Article Classification

## Video Presentation: [Click here](https://drive.google.com/file/d/1rZ5yQgPggodkhZzjGuz6R7Kjjs7hgTaJ/view?usp=drive_link)


```python
news_data = pd.read_csv("data_news.csv")
news_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>links</th>
      <th>short_description</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WELLNESS</td>
      <td>143 Miles in 35 Days: Lessons Learned</td>
      <td>https://www.huffingtonpost.com/entry/running-l...</td>
      <td>Resting is part of training. I've confirmed wh...</td>
      <td>running-lessons</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WELLNESS</td>
      <td>Talking to Yourself: Crazy or Crazy Helpful?</td>
      <td>https://www.huffingtonpost.com/entry/talking-t...</td>
      <td>Think of talking to yourself as a tool to coac...</td>
      <td>talking-to-yourself-crazy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WELLNESS</td>
      <td>Crenezumab: Trial Will Gauge Whether Alzheimer...</td>
      <td>https://www.huffingtonpost.com/entry/crenezuma...</td>
      <td>The clock is ticking for the United States to ...</td>
      <td>crenezumab-alzheimers-disease-drug</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Data Collection and Preprocessing

### a. Missing data


```python
news_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 5 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   category           50000 non-null  object
     1   headline           50000 non-null  object
     2   links              50000 non-null  object
     3   short_description  50000 non-null  object
     4   keywords           47332 non-null  object
    dtypes: object(5)
    memory usage: 1.9+ MB
    

- 50,000 rows with 2668 values missing in keywords column.

### b. Unique news


```python
news_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>links</th>
      <th>short_description</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>50000</td>
      <td>47332</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10</td>
      <td>45577</td>
      <td>45745</td>
      <td>45743</td>
      <td>41558</td>
    </tr>
    <tr>
      <th>top</th>
      <td>WELLNESS</td>
      <td>Sunday Roundup</td>
      <td>https://www.huffingtonpost.com/entry/bryce-har...</td>
      <td>Along with his fists, the star Nationals outfi...</td>
      <td>post</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5000</td>
      <td>22</td>
      <td>8</td>
      <td>8</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>



- Can avoid links column.
- There are 10 unique categories.
- There are 45577 unique headlines and 45743 unique short descriptions. If a short description is repeated, it is highly likely a repeat entry.
- Keywords can be repeated because different articles can have same keywords.

### c. Category distribution


```python
news_data["category"].value_counts()
```




    category
    WELLNESS          5000
    POLITICS          5000
    ENTERTAINMENT     5000
    TRAVEL            5000
    STYLE & BEAUTY    5000
    PARENTING         5000
    FOOD & DRINK      5000
    WORLD NEWS        5000
    BUSINESS          5000
    SPORTS            5000
    Name: count, dtype: int64



- All the categories are equally distributed with 5000 each.

### d. Remove the link column

- Since the link does not contribute to finding categories.


```python
required_data = news_data.drop('links', axis=1)
```

### e. Check headline, description and keyword length


```python
sns.set_style("whitegrid")

fig, axes = plt.subplots(3, 2, figsize=(8, 12))

sns.boxplot(required_data['headline'].apply(len), ax=axes[0,0])
axes[0,0].set_title("Headline Length Outliers")

sns.histplot(required_data['headline'].apply(len), bins=50, kde=True, ax=axes[0,1])
axes[0,1].set_title("Headline Length Distribution")

sns.boxplot(required_data['short_description'].apply(len), ax=axes[1,0])
axes[1,0].set_title("Short Description Length Outliers")

sns.histplot(required_data['short_description'].apply(len), bins=50, kde=True, ax=axes[1,1])
axes[1,1].set_title("Short Description Length Distribution")

keyword_length = required_data['keywords'].fillna('').apply(len)

sns.boxplot(keyword_length, ax=axes[2,0])
axes[2,0].set_title("Keyword Length Outliers")

sns.histplot(keyword_length, bins=50, kde=True, ax=axes[2,1])
axes[2,1].set_title("Keyword Length Distribution")

plt.tight_layout()
plt.show()
```


    
![png](README_Resources/output_95_0.png)
    


- Many outliers in headline, short descriptions and keywords.
- Headline distribution looks balanced.
- Short description is extremely uneven.
- Keywords look normally distributed with a positive skew.

### f. Keep only data with unique short descriptions


```python
required_data = required_data.drop_duplicates(subset='short_description', keep='first')
required_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>short_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45743</td>
      <td>45743</td>
      <td>45743</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10</td>
      <td>45571</td>
      <td>45743</td>
    </tr>
    <tr>
      <th>top</th>
      <td>WELLNESS</td>
      <td>Sunday Roundup</td>
      <td>Resting is part of training. I've confirmed wh...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5000</td>
      <td>22</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- After keeping only unique short descriptions, 4257 rows were removed

### g. Audit repeated headlines

- Headlines can be same but description is different, eg: Sunday Roundup.
- Keywords and headline is similar, cleaning and tokenization will make them similar, therefore, it is better to remove the keywords 


```python
required_data[required_data['headline'].duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>short_description</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3753</th>
      <td>WELLNESS</td>
      <td>The Moment I Knew</td>
      <td>It's the illogical shame you feel when you tak...</td>
      <td>the-moment-i-knew</td>
    </tr>
    <tr>
      <th>4257</th>
      <td>WELLNESS</td>
      <td>Never Too Late</td>
      <td>The next time you think about grabbing a bag o...</td>
      <td>weight-loss-success</td>
    </tr>
    <tr>
      <th>5324</th>
      <td>POLITICS</td>
      <td>Sunday Roundup</td>
      <td>Happy 4th of July weekend! The week leading up...</td>
      <td>sunday-roundup</td>
    </tr>
    <tr>
      <th>5843</th>
      <td>POLITICS</td>
      <td>Sunday Roundup</td>
      <td>This week proved that while the arc of the mor...</td>
      <td>sunday-roundup</td>
    </tr>
    <tr>
      <th>5932</th>
      <td>POLITICS</td>
      <td>Sunday Roundup</td>
      <td>This week, the House voted along party lines t...</td>
      <td>sunday-roundup</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39549</th>
      <td>WORLD NEWS</td>
      <td>North Korea Fires Submarine-Launched Missile: ...</td>
      <td>North Korea has been testing its weapons syste...</td>
      <td>north-korea-submarine-missile</td>
    </tr>
    <tr>
      <th>39863</th>
      <td>WORLD NEWS</td>
      <td>The Crossing</td>
      <td>An immersive reporting series hosted by Susan ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41420</th>
      <td>BUSINESS</td>
      <td>Redefining Success</td>
      <td>Success is your net-worth. Your net-worth not ...</td>
      <td>redefining-success</td>
    </tr>
    <tr>
      <th>42075</th>
      <td>BUSINESS</td>
      <td>10 Ways To Spot A Truly Exceptional Employee</td>
      <td>Dealing with difficult people is frustrating a...</td>
      <td>10-ways-to-spot-a-truly-e</td>
    </tr>
    <tr>
      <th>44032</th>
      <td>BUSINESS</td>
      <td>The 9 Worst Mistakes You Can Ever Make At Work</td>
      <td>Self-awareness is a critical skill in the work...</td>
      <td>the-9-worst-mistakes-you-can-ever-make-at-work</td>
    </tr>
  </tbody>
</table>
<p>172 rows × 4 columns</p>
</div>



### h. Removing keywords column


```python
required_data = required_data.drop('keywords', axis=1)
```

### i. Remove short description and headline length outliers


```python
# Calculate Q1 and Q3
Q1 = required_data['short_description'].apply(len).quantile(0.25)
Q3 = required_data['short_description'].apply(len).quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
required_data = required_data[(required_data['short_description'].apply(len) >= lower_bound) & 
                                 (required_data['short_description'].apply(len) <= upper_bound)
]
```


```python
# Calculate Q1 and Q3
Q1 = required_data['headline'].apply(len).quantile(0.25)
Q3 = required_data['headline'].apply(len).quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
required_data = required_data[(required_data['headline'].apply(len) >= lower_bound) & 
                                 (required_data['headline'].apply(len) <= upper_bound)
]
```


```python
fig, axes = plt.subplots(2, 2, figsize=(8, 12))

sns.boxplot(required_data['headline'].apply(len), ax=axes[0,0])
axes[0,0].set_title("Headline Length Outliers")

sns.histplot(required_data['headline'].apply(len), bins=50, kde=True, ax=axes[0,1])
axes[0,1].set_title("Headline Length Distribution")

sns.boxplot(required_data['short_description'].apply(len), ax=axes[1,0])
axes[1,0].set_title("Short Description Length Outliers")

sns.histplot(required_data['short_description'].apply(len), bins=50, kde=True, ax=axes[1,1])
axes[1,1].set_title("Short Description Length Distribution")

plt.tight_layout()
plt.show()
```


    
![png](README_Resources/output_108_0.png)
    


- Short description has most character lengths between 120 and 130.


```python
required_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>short_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>44429</td>
      <td>44429</td>
      <td>44429</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10</td>
      <td>44279</td>
      <td>44429</td>
    </tr>
    <tr>
      <th>top</th>
      <td>FOOD &amp; DRINK</td>
      <td>Weekly Roundup of eBay Vintage Clothing Finds ...</td>
      <td>Resting is part of training. I've confirmed wh...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4973</td>
      <td>17</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- After cleanup, 5571 rows are removed


```python
required_data["category"].value_counts()
```




    category
    FOOD & DRINK      4973
    STYLE & BEAUTY    4939
    ENTERTAINMENT     4899
    WELLNESS          4862
    PARENTING         4836
    TRAVEL            4798
    POLITICS          4782
    WORLD NEWS        4769
    BUSINESS          2948
    SPORTS            2623
    Name: count, dtype: int64



- The balance of categories is affected. Business and Sports articles are poorly represented compared to others.

### j. Concatenate headline and description for preprocessing


```python
article_cleaned_data = pd.DataFrame({
    'combined_text': required_data['headline'] + ' ' + required_data['short_description'],
    'category': required_data['category']
})
```

### k. Lowercase the text


```python
article_cleaned_data['processed_text'] = article_cleaned_data['combined_text'].apply(
    lambda x: x.lower()
)
```

### l. Remove HTML tags


```python
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: re.sub('<.*?>', '', x)
)
```

### m. Remove punctuations and special characters


```python
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: re.sub(r'[^\w\s]', '', x)
)
```

### n. Tokenize the text


```python
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: word_tokenize(x)
)
```

### o. Remove stop words


```python
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: [word for word in x if word not in stop_words]
)
```

### p. Lemmatize the tokens


```python
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)
```

### q. Remove numbers and words having 2 characters or less


```python
# Remove the numbers and words, 2 characters or less
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: [re.sub(r'\b\w{1,2}\b|\d+', '', word) for word in x]
)

# Get rid of empty strings
article_cleaned_data['processed_text'] = article_cleaned_data['processed_text'].apply(
    lambda x: [word for word in x if word != '']
)
```

## 2. Feature Extraction

### a. Distribution of different categories


```python
# Word frequency analysis
food_drink = article_cleaned_data[article_cleaned_data['category'] == 'FOOD & DRINK']['processed_text']
style_beauty = article_cleaned_data[article_cleaned_data['category'] == 'STYLE & BEAUTY']['processed_text']
entertainment = article_cleaned_data[article_cleaned_data['category'] == 'ENTERTAINMENT']['processed_text']
wellness = article_cleaned_data[article_cleaned_data['category'] == 'WELLNESS']['processed_text']
parenting = article_cleaned_data[article_cleaned_data['category'] == 'PARENTING']['processed_text']
travel = article_cleaned_data[article_cleaned_data['category'] == 'TRAVEL']['processed_text']
politics = article_cleaned_data[article_cleaned_data['category'] == 'POLITICS']['processed_text']
world_news = article_cleaned_data[article_cleaned_data['category'] == 'WORLD NEWS']['processed_text']
business = article_cleaned_data[article_cleaned_data['category'] == 'BUSINESS']['processed_text']
sports = article_cleaned_data[article_cleaned_data['category'] == 'SPORTS']['processed_text']

# Join all lists of words into a single string
text_food_drink = ' '.join([word for review in food_drink for word in review])
text_style_beauty = ' '.join([word for review in style_beauty for word in review])
text_ent = ' '.join([word for review in entertainment for word in review])
text_well = ' '.join([word for review in wellness for word in review])
text_parent = ' '.join([word for review in parenting for word in review])
text_travel = ' '.join([word for review in travel for word in review])
text_polit = ' '.join([word for review in politics for word in review])
text_world = ' '.join([word for review in world_news for word in review])
text_business = ' '.join([word for review in business for word in review])
text_sports = ' '.join([word for review in sports for word in review])

# Create a figure with 10 subplots
fig, axs = plt.subplots(5, 2, figsize=(16, 12))

# Create a word cloud object
wordcloud_food_drink = WordCloud(width=800, height=400).generate(text_food_drink)
wordcloud_style_beauty = WordCloud(width=800, height=400).generate(text_style_beauty)
wordcloud_ent = WordCloud(width=800, height=400).generate(text_ent)
wordcloud_well = WordCloud(width=800, height=400).generate(text_well)
wordcloud_parent = WordCloud(width=800, height=400).generate(text_parent)
wordcloud_travel = WordCloud(width=800, height=400).generate(text_travel)
wordcloud_polit = WordCloud(width=800, height=400).generate(text_polit)
wordcloud_world = WordCloud(width=800, height=400).generate(text_world)
wordcloud_business = WordCloud(width=800, height=400).generate(text_business)
wordcloud_sports = WordCloud(width=800, height=400).generate(text_sports)

# Plot the word cloud for positive reviews on the first subplot
axs[0,0].imshow(wordcloud_food_drink, interpolation='bilinear')
axs[0,0].set_title('Food and Drink')
axs[0,0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[0,1].imshow(wordcloud_style_beauty, interpolation='bilinear')
axs[0,1].set_title('Style and Beauty')
axs[0,1].axis('off')

# Plot the word cloud for positive reviews on the first subplot
axs[1,0].imshow(wordcloud_ent, interpolation='bilinear')
axs[1,0].set_title('Entertainment')
axs[1,0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[1,1].imshow(wordcloud_well, interpolation='bilinear')
axs[1,1].set_title('Wellness')
axs[1,1].axis('off')

# Plot the word cloud for positive reviews on the first subplot
axs[2,0].imshow(wordcloud_parent, interpolation='bilinear')
axs[2,0].set_title('Parenting')
axs[2,0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[2,1].imshow(wordcloud_travel, interpolation='bilinear')
axs[2,1].set_title('Travel')
axs[2,1].axis('off')

# Plot the word cloud for positive reviews on the first subplot
axs[3,0].imshow(wordcloud_polit, interpolation='bilinear')
axs[3,0].set_title('Politics')
axs[3,0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[3,1].imshow(wordcloud_world, interpolation='bilinear')
axs[3,1].set_title('World News')
axs[3,1].axis('off')

# Plot the word cloud for positive reviews on the first subplot
axs[4,0].imshow(wordcloud_business, interpolation='bilinear')
axs[4,0].set_title('Business')
axs[4,0].axis('off')

# Plot the word cloud for negative reviews on the second subplot
axs[4,1].imshow(wordcloud_sports, interpolation='bilinear')
axs[4,1].set_title('Sports')
axs[4,1].axis('off')

# Layout so plots do not overlap
fig.tight_layout()

plt.show()
```


    
![png](README_Resources/output_132_0.png)
    


- Food and drink have words like recipe, make, etc.
- Style and Beauty have words like photo, dress, fashion, etc.
- Entertainment have words like film, show, movie, etc.
- Wellmess have words like life, people, way, etc.
- Parenting have words like child, parent, kid, etc.
- Travel have words like photo, travel, hotel, etc.
- Politics have words like state, say, bill, etc.
- World news have words like attack, country, new, etc.
- Business have words like company, business, job, etc.
- Sports have words like game, team, win etc.

### b. Convert text data to numbers using TF-IDF for model use


```python
# Join words for vectorization
article_cleaned_data['processed_tokens'] = article_cleaned_data['processed_text'].apply(
    lambda x: ' '.join(x)
)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(article_cleaned_data['processed_tokens'])
```

## 3. Model Development and Training

### a. Encode categorical variable


```python
for column in ['category']:
    le = LabelEncoder()
    article_cleaned_data[column] = le.fit_transform(article_cleaned_data[column])
```

### b. Split to train and test set


```python
# Prepare data
X = tfidf_features
y = article_cleaned_data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### c. Train Logistic Regression


```python
# Train logistic regression model
lr_news = LogisticRegression(random_state=42)
lr_news.fit(X_train, y_train)
# Predict labels on the test set
y_pred = lr_news.predict(X_test)
```

#### c1. Evaluate Logistic regression


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()
```

    Accuracy: 0.7970965563808238
    Precision: 0.7990204823674877
    Recall: 0.7970965563808238
    F1 Score: 0.796620355919033
    ROC-AUC: 0.86353378745356
    


    
![png](README_Resources/output_144_1.png)
    


### d. Train Naive Bayes


```python
# Train the Naive Bayes classifier
nb_news_model = MultinomialNB()
nb_news_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_news_model.predict(X_test)
```

#### d.1. Evaluate Naive Bayes


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.show()
```

    Accuracy: 0.7587215845149674
    Precision: 0.778732082301321
    Recall: 0.7587215845149674
    F1 Score: 0.749211640617267
    ROC-AUC: 0.86353378745356
    


    
![png](README_Resources/output_148_1.png)
    


### e. Train SVM


```python
# Create and train model
svm_news = SVC(random_state=42)
svm_news.fit(X_train, y_train)

# Make predictions
y_pred = svm_news.predict(X_test)
```

#### e.1. Evaluate SVM


```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.show()
```

    Accuracy: 0.795295971190637
    Precision: 0.7993565057439405
    Recall: 0.795295971190637
    F1 Score: 0.7953872693835766
    ROC-AUC: 0.86353378745356
    


    
![png](README_Resources/output_152_1.png)
    


### f. Randomized search for hyper parameter tuning

- Performed tuning on logistic regression since it performed slightly better than SVM


```python
# Logistic Regression model
logreg = LogisticRegression(random_state=42)

# Hyperparameter space
param_distributions = {
    'C': uniform(loc=0.01, scale=10),  # Range of C (e.g., from 0.01 to 10)
    'penalty': ['l2'],         # Regularization types
    'solver': ['liblinear'],  # Solvers supported for given penalties
}

# Random search setup
random_search = RandomizedSearchCV(
    estimator=logreg,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    scoring='f1_weighted',  # Metric to optimize
    random_state=42,
    verbose=1
)

# Fit on training data
random_search.fit(X_train, y_train)

# Best parameters and performance
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best F1 Score: {random_search.best_score_:.2f}")
```

    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    Best Parameters: {'C': 3.7554011884736247, 'penalty': 'l2', 'solver': 'liblinear'}
    Best F1 Score: 0.80
    

### g. Best Model


```python
# Use best estimator for predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic regression Confusion Matrix")
plt.show()
```

    Accuracy: 0.8032860679720909
    Precision: 0.8039221796184353
    Recall: 0.8032860679720909
    F1 Score: 0.8029854405956157
    ROC-AUC: 0.86353378745356
    


    
![png](README_Resources/output_157_1.png)
    


## 4. Model Evaluation

- The best model is logistic regression model with f1 score: 0.802
- Precision and Recall are balanced at 0.803
- Accuracy is also at 0.80
- Naive bayes is the worst with 0.74 f1 score.
- SVM is slightly poor with 0.795 f1 score and takes a long time to train.
