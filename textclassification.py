import pandas as pd

# Try reading the CSV file with different encodings
encodings = ["utf-8", "latin1", "cp1252"]

for encoding in encodings:
    try:
        data = pd.read_csv('articles.csv', encoding=encoding)
        break  # If successful, exit the loop
    except UnicodeDecodeError:
        continue

# List of columns to remove
columns_to_remove = ["Id", "Article.Banner.Image", "Outlets", "Tonality"]

# Drop the specified columns from the DataFrame
df = data.drop(columns=columns_to_remove)

# Initialize a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Calculate TF-IDF features for both columns
tfidf_matrix = vectorizer.fit_transform(df["Article.Description"] + " " + df["Full_Article"])

# Initialize a variable to accumulate the total similarity score
total_similarity = 0

# Iterate through each row and compare it to all other rows
for i in range(len(df)):
    for j in range(len(df)):
        # Calculate cosine similarity for the pair of rows
        cosine_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
        
        # Add the similarity score to the total
        total_similarity += cosine_sim[0, 0]

# Calculate the average similarity by dividing the total by the number of comparisons
num_comparisons = len(df) ** 2
average_similarity = total_similarity / num_comparisons

print("Average Cosine Similarity Score:", average_similarity)

# Remove the "Full_Article" column
if average_similarity>0.9:
    if 'Full_Article' in df.columns:
        df = df.drop('Full_Article', axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv('dataset.csv', index=False)

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)  # Specify num_classes

# Load your dataset from a CSV file
df = pd.read_csv(dataset.csv)

# Extract the relevant columns from your dataset
train_data = df["Heading"].tolist()  # Replace "Heading" with your actual column name
test_data = df["Article.Description"].tolist()  # Replace "Article.Description" with your actual column name
labels = df["Article_Type"].tolist()  # Assuming you have a column named "Article_Type"

# Use label encoding to convert string labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize your data
train_encodings = tokenizer(train_data, truncation=True, padding=True)
test_encodings = tokenizer(test_data, truncation=True, padding=True)

# Convert data to PyTorch tensors
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(labels)  # Now labels are numeric

# Create a DataLoader for training data
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(3):  # Number of training epochs
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("./bert_classification_model")
