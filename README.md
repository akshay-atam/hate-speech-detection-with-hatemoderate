# Casting Light on Dark Words - Hate Speech Detection using Hate Moderate
In today's digital era, hate speech is a growing concern in online spaces. This NLP project aims to improve hate speech detection by simplifying the categorization process. The main goal is to accurately identify and categorize hate speech by utilizing Facebook community guidelines and OpenAI’s ‘moderations’ API.

There are a plethora of existing hate speech datasets and state-of-the-art models that utilize those datasets and predict the accuracy of detecting hate speech in texts. However, existing datasets have an underlying bias of having examples from certain online sources (even manually crafted) and, most importantly, restricted to certain types of hate speech. Benchmark datasets such as ETHOS or HateXplain, classify hate speech based on actual hate, offensive, or neutral. Such bias poses a limit on how well any model can be trained. Thus, there is a need of a homogenous and diverse dataset which covers the various classes of hate speech. 

In this project, I am driven by the following Research Questions:

## RQ1: How do deep neural networks perform against rule based natural language policies?
To answer this question, I would use HateModerate, a custom dataset with hate speech examples following the Facebook Community Guidelines. The dataset aims to solve the underlying bias by having 41 different categories of hate speech. Following is the generalized description of each tier:

1. Content that targets individuals or groups based on protected characteristics with violent or dehumanizing speech, including harmful stereotypes, denial of existence, and mockery of hate crimes.
2. Content that targets individuals or groups based on protected characteristics with generalizations of inferiority, including derogatory terms related to hygiene, appearance, mental health, and moral character, as well as expressions of contempt, hate, dismissal, and disgust, excluding certain gender-based cursing in a romantic context.
3. Content targeting individuals or groups based on protected characteristics with calls for segregation, explicit exclusion, political exclusion, economic exclusion, and social exclusion, except for gender-based exclusion in health and positive support groups.
4. Content that describes or negatively targets people with slurs, defined as words creating an atmosphere of exclusion and intimidation based on protected characteristics, even when targeting individuals outside the group that the slur inherently addresses.

For this project, I would be simplifying the classification by combining the 41 categories to just 4 by classifying them according to the tiers mentioned above. 

## RQ2: How do deep neural networks perform against rule based natural language policies?
To answer this question, I would be using pre-trained language model and OpenAI’s moderations API to classify hate speech according to Facebook community guidelines. First, I would use the pre-trained language model to test the accuracy of detecting hate speech. Next, using OpenAI’s moderations API, I would see how examples created from one guideline perform with other the rules of other guidelines. 

To address these research questions, I conducted Exploratory Data Analysis (EDA) to gain valuable insights into the characteristics of the hate speech examples within the dataset. The next step was to use a pre-trained language model to assess the accuracy of detecting hate speech. I used DistilBERT as my language model as it gives a sweet spot between model size and performance. 

After the training of pre-trained language model, I used OpenAI’s moderations API to find confidence scores of hate speech examples according to the categories defined by OpenAI. This metric would create an amalgamation of two distinct guidelines and will give us better understanding of the generalization strength of the dataset.

Finally, to test the integration of Facebook and OpenAI’s guidelines, I would use the confidence scores obtained by the moderations API to train an ensemble model for classifying hate speech. The metrics evaluated by this model will let us know the effectiveness of the integration and find potential areas for improvement.

To accomplish these objectives, I adopt a three-pronged strategy to evaluate an address hate speech. 

## Prong 1: DistilBERT pre-trained language model
I utilize the pre-trained language model, DistilBERT, to train it on hate speech examples. The accuracy of the API is measured against actual tier labels, and the breakdown of policy violation flags across categories is examined.

## Prong 2: OpenAI’s ‘moderations’ API
I utilize the OpenAI Moderations API to assess hate speech examples categorized into four tiers per the Facebook community guidelines. The accuracy of the API is measured against actual tier labels, and the breakdown of policy violation flags across categories is examined.

## Prong 3: Machine Learning Model
I utilize the ‘category_scores’ provided by the API call as features for training a machine learning model. The scores indicate the model's confidence in policy violation for specific categories (of OpenAI). 
The reason to adopt this three-pronged approach is to gain valuable insights from the performance of pre-trained language models and direct API classification with a machine learning model for improving hate speech moderation accuracy and finding areas for improvement.

# Results
Following are the results obtained from each experiment:
## DistilBERT
Trained for 10 epochs, using CrossEntropyLoss() as loss function yielded,
- Accuracy: 0.9198
- Precision: 0.9210
- Recall: 0.9198
- F1-score: 0.9201
As we can see, DistilBERT performed exceptionally well on hate speech examples. All the metrics are very close to one oanother. The training time was also impressive, taking only 12 minutes to train 5,430 examples on Nvidia V100 GPU.
## OpenAI moderations API
The result was calculated based on the number of examples OpenAI flagged as True. Without considering the respective classes, the API flagged 4,558 examples as hate speech, achieving an accuracy of 0.8394 which is quite impressive. Upon checking the hate speech examples labelled as false, it turned out that examples from tier 3 were not classified as hate speech possibly because tier 3 considers hate speech based on exclusion factors (social, economic, and others), a category that was present in Facebook Community Guidelines but not in OpenAI’s.
## Ensemble model
Accuracy: 0.6869
The ensemble model had the least performance out of the three. This performance can be seen in other metrics as well. Below table showcases the Precision, Recall and F1-score of all the four tiers.
| | Tier 1| Tier 2 | Tier 3 | Tier 4 |
| --- | --- | --- | --- | --- |
|Precision | 0.69 | 0.67 | 0.76 | 0.67 |
|Recall	|0.62 | 0.85 | 0.46 | 0.4 |
|F1-score | 0.65 | 0.75 | 0.58 | 0.5 |

As we can see, tier 2 has the highest metrics overall while tier 4 has the worst performance overall. This closely aligns with the number of examples present in each tier, hence, hinting at the imbalance is big enough that the ensemble model cannot rectify that.


