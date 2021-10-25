# Disaster Message Classifier
#### Improving Access to Critical Information During Crisis

<img src="imgs/Intro.jpeg" alt="Drawing" style="width: 400px;"/>

 ## 3. File Description
~~~~~~~
        disaster_response_pipeline
          |-- app                            
                |-- app.py                   
          |-- datasets
                |-- df_clean.csv    
                |-- disaster_response_messages_test.csv   
                |-- disaster_response_messages_training.csv     
                |-- disaster_response_messages_validation.csv
          |-- models
                |-- multiout_adaboost.pkl
          |-- notebooks
                |-- 1_Data_cleaning.ipynb
                |-- 2_EDA.ipynb
                |-- 3_Random_Forest_Spellcheck.ipynb
                |-- 4_Decision_Tree_Adaboost-SpellCheck.ipynb
          |-- reports
                |-- Baseline_all.csv
                |-- Classification_reports
                |-- ROC_AUC_reports
          |-- README.md
          |-- Presentation.pdf
          |-- Executive_Summary.pdf
~~~~~~~
### Table of Contents

1. [Project Overview](#ProjectOverview)
2. [Dataset Description](#Dataset)
3. [Machine Learning process](#MLprocess)
4. [Software Dependencies](#Software)
5. [ML Findings](#MLFindings)
6. [Model Evaluation](#ModelEval)
7. [Conclusion](#Conclusion)
8. [Next Steps](#NextSteps)
9. [App Development](#AppDevelopment)

<a name="ProjectOverview"></a>
## 1. Project Overview
Natural disasters affect almost every part of the world. In 2018, Indonesia faced the highest number of deaths in the world due to the earthquakes and tsunami that occurred in September. In the United States that year, most of fatalities from natural disasters came from tropical cyclones, wildfires, heat, and drought.<br><br>
**Problem at hand**<br>
Social media is being explored as tool for disaster management by developers, researchers, government agencies and businesses. The disaster-affected area requires both cautionary and disciplinary measures. The need for decision-making system during emergencies and in real time poses problems classifying emergencies. <br><br>
>**Problem Statement**<br>
Can we explore Social Media tools to better classify emergency messages during times of crisis to help people make more informed and better decisions?


<a name="Dataset"></a>
## 2. Dataset Description

This dataset available at [Kaggle](https://www.kaggle.com/landlord/multilingual-disaster-response-messages/) provided by Figure Eight, contains 30,000 messages drawn from events of earthquakes, floods, hurricanes and news articles spanning a large number of years and 100s of different disasters. Original message and it's English translation is provided.

The data has been encoded with 36 different categories related to disaster response. These classes are noted in column titles already binarized.

#### `Baseline` for each 36 class labels:
><img src="imgs/Baseline1.png" alt="Drawing" style="width: 800px;"/>
><img src="imgs/Baseline2.png" alt="Drawing" style="width: 600px;"/>

Tough Baseline due to inbalanced data, because negative class contains more than 90% of data in most categories.

<a name="MLprocess"></a>
## 3. Machine Learning Pipeline

* **Random Forest Classifier**: Had the best Model performance, creating a machine learning pipeline, using conventional NLTK, tokenizing and lemmatizing, and multi-output classification to output all classes that apply.
 - The **differential** here was using stopwords and spelling check. Since at times of crisis, mispellings are more common. However running your model with spellcheck takes 3 hours more than standard tokenizing (which takes 5min).<br><br>
 - **Multioutput Classifier**: Consists of fitting multiple classifiers for each target to generate a multi-output data. 

<a name="Software"></a>
## 4. Software Dependencies

This project uses Python 3.7.2 and the following libraries:
* Python 3.0+
* [NumPy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/stable/) 
* [NLTK](https://www.nltk.org/)
* [TextBlob](https://textblob.readthedocs.io/en/dev/)
* [Pickle](https://docs.python.org/3/library/pickle.html)
* [Streamlit](https://docs.streamlit.io/library/api-reference)

<a name="MLFindings"></a>
## 5. Machine Learning Pipeline

Dataset is highly imbalanced and that is the reason why the accuracy is high and Recall is considerably low. In imbalanced data we should focus on F1 Score, not in accuracy. I'm focusing on catching critical messages, that said, false negatives are more critical to be evaluated. In order to get a better model from this perspective, recall will provide an indication of missed positive disaster messages. 


|                        | Precision | Recall | Accuracy | F1-Score | Support |
|------------------------|-----------|--------|----------|----------|---------|
| <font color='red'>Request</font>                | 0.86      | 0.40   | 0.88     | 0.55     | 1022    |
| Offer                  | 0.00      | 0.00   | 0.99     | 0.00     | 30      |
| Aid Related            | 0.81      | 0.61   | 0.76     | 0.69     | 2529    |
| Medical Help           | 0.76      | 0.04   | 0.91     | 0.07     | 527     |
| Search and Rescue      | 0.62      | 0.03   | 0.97     | 0.06     | 169     |
| Security               | 0.00      | 0.00   | 0.98     | 0.00     | 102     |
| Military               | 0.83      | 0.03   | 0.96     | 0.05     | 189     |
| <font color='red'>Water</font>                  | 0.89      | 0.20   | 1.0      | 0.32     | 388     |
| <font color='red'>Food</font>                   | 0.87      | 0.51   | 0.93     | 0.64     | 639     |
| Shelter                | 0.85      | 0.24   | 0.92     | 0.37     | 535     |
| Clothing               | 0.80      | 0.05   | 0.98     | 0.09     | 82      |
| Money                  | 0.60      | 0.02   | 0.97     | 0.04     | 160     |
| Missing People         | 0.00      | 0.00   | 0.98     | 0.00     | 69      |
| Refugees               | 0.69      | 0.06   | 0.96     | 0.10     | 199     |
| <font color='red'>Death</font>                  | 0.86      | 0.1    | 0.95     | 0.23     | 279     |
| Other Aid              | 0.80      | 0.01   | 0.86     | 0.0      | 810     |
| Infrastructure Related | 0.00      | 0.00   | 0.93     | 0.00     | 377     |
| Transport              | 0.65      | 0.09   | 0.95     | 0.17     | 296     |
| Buildings              | 0.75      | 0.04   | 0.94     | 0.07     | 319     |
| Electricity            | 0.75      | 0.05   | 0.98     | 0.09     | 120     |
| Tools                  | 0.00      | 0.00   | 0.99     | 0.00     | 36      |
| Hospitals              | 0.00      | 0.00   | 0.98     | 0.00     | 70      |
| Shops                  | 0.00      | 0.00   | 0.99     | 0.00     | 20      |
| Aid Centers            | 0.00      | 0.00   | 0.98     | 0.00     | 75      |
| Other Infrastructure   | 0.00      | 0.00   | 0.95     | 0.96     | 243     |
| Weather Related        | 0.89      | 0.64   | 0.87     | 0.74     | 1701    |
| <font color='red'>Floods</font>                 | 0.95      | 0.37   | 0.94     | 0.53     | 504     |
| <font color='red'>Fire</font>                   | 1.00      | 0.02   | 0.99     | 0.03     | 59      |
| <font color='red'>Storm</font>                  | 0.82      | 0.48   | 0.94     | 0.61     | 541    |
| <font color='red'>Earthquake</font>             | 0.91      | 0.80   | 0.97     | 0.85     | 570     |
| Cold                   | 0.91      | 0.07   | 0.97     | 0.14     | 137     |
| Other Weather          | 0.50      | 0.01   | 0.94     | 0.03     | 341     |
| Direct Report          | 0.88      | 0.34   | 0.86     | 0.49     | 1161    |
|                        |           |        |          |          |         |
| Micro Average          | 0.85      | 0.4    |          | 0.49     | 14608   |
| Macro Average          | 0.58      | 0.15   |          | 0.20     | 14608   |
| Weighted Average       | 0.77      | 0.34   |          | 0.43     | 14608   |
| Samples Average        | 0.40      | 0.21   |          | 0.26     | 14608   |

<a name="ModelEval"></a>
## 6. Model Evaluation
I chose evaluating for micro average, since my data is pretty unbalanced the micro-average will adequately capture this class imbalance, and bring the overall precision average down. And although my Decision Tree has a better Micro Average F1 Score, Random Forest Performed better in some other features. I would pick the appropriate model according to your preference of category.  Dataset is highly imbalanced and that is the reason why the accuracy is high and Recall is considerably low. In imbalanced data we should focus on F1 Score (weighted average of the precision and recall), not in accuracy. I'm focusing on catching critical messages, that said, false negatives are more critical to be evaluated. 
We do not want miss critical food or water helps, or rescue helps. In order to get a better model from this perspective, recall will be more useful for us.

<img src="imgs/Model evaluation.png" alt="Drawing" style="width: 800px;"/>

<a name="Conclusion"></a>
 ## 7. Conclusion
 - Providing contextualized and timely information during times of crisis helps people make more informed and better decisions.
 - When we are looking to crisis questions or other humanitarian large scale problems. There is an opportunity to collaborate in a global scale.
 - Social media platforms can be efficiently used for supply chain management by professionals, organizations and readers for their operations. 


<a name="NextSteps"></a>
 ## 8. Next Steps
 
 - Reattempt to retry Resampling, under-sampling and BalancedBaggingClassifier to normalize data.
 - Reattempt for class_weight hyper-parameters by putting less weights on the majority class intances. 
 - Aggregate realtime twitter messages for temporal location information.
 - Aggregate to Google Earth Engine to identify and outline public satellite image to illustrate disaster affected area.
 - Combine them together


<a name="AppDevelopment"></a>
 ## 9. App Development
 
 <img src="imgs/app_video.gif" alt="Drawing" style="width: 800px;"/>
 

 
 
 
 
 
 
# disasermessageapp
