# Data analysis from .csv file#
Columns required for the prediction of the article type

Heading: The heading or title of an article often contains important keywords and information that can help identify the topic or subject matter, which is relevant for classifying the article type.

Article.Description: The article description can provide additional context and content, making it valuable for classifying articles into specific types.

Columns not required for the prediction of the article type

1. Id : This column serves as a unique identifier for each article but does not contain information about the content or characteristics of the articles that would be useful for the classification task.

2. Article.Banner.Image: This column likely contains links to images associated with the articles. It is typically unrelated to the textual content that is essential for text classification.

3. Outlets: This column appears to represent the source or publication outlets. While it may provide information about the source of the article, it is not directly related to the content that determines article type.

4. Tonality: The "Tonality" column relates to the emotional tone of the information, which may be interesting for sentiment analysis but is generally not a direct indicator of article type.

#Removal of duplicate information
 calculate the cosine similarity between the column Article.Description,Full_Article. Based on the average similarity, Full_Article column can be removed

 #Final data for training
 Then we can use the final data saved as dataset.csv can be used for the Vectorizatoin and training purpose
