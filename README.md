# TextualTherapists at MentalRiskES-IberLEF2023: Early Detection of Depression using a User-level Feature-based Machine Learning Approach

<center>
  <img src="https://cdn2.iconfinder.com/data/icons/coronavirus-10/512/headache-pain-head-sick-sickness-512.png" width="280" height="300">
</center>

__System developed by the TextualTherapists team to participate in Task 2: Depression detection of MentalRiskES@IberLEF2023__

# Authors

* __Alberto Fernández Hernández__
* __Raúl Moreno Sánchez__
* __José Viosca Ros__
* __Raquel Enrique Guillén__
* __Noa P. Cruz-Díaz__
* __Salud María Jiménez-Zafra__

# Summary

This project presents the participation of the TextualTherapists team in the MentalRiskES shared task at the IberLEF 2023 evaluation campaign, focusing on the __early risk prediction of mental disorders in Spanish__: detecting whether a user suffers from depression or not based on a set of comments posted on Telegram. We addressed this task using several machine learning approaches that integrates lexical, sentiment, toxicity, and emotional features and that takes into account the PHQ9 Patient Questionnaire.

# Main goals

* Detect depressed users
* Moreover, detect depressed users as early as possible

# Approaches

To do so, several approaches are used:

* __Traditional Machine Learning approach__: in order to avoid outrageous huge transformers models (with millions of parameters), we explore a simplier machine learning approach, selecting carefully the variables:
   * PHQ-9 terms, from Patient Health Questionnarie 
   * Part of Speech (POS) features 
   * Sentiment features: positive, negative, neutral; extracted from several pre-trained models
     * [__RoBERTa base sentiment__](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
     * [__TextBlob library__](https://textblob.readthedocs.io/en/dev/)
     * [__VADER library__](https://github.com/cjhutto/vaderSentiment)
     * [__Emojis sentiments__](https://kt.ijs.si/data/Emoji_sentiment_ranking/)
   * Emotion features
     * [__NRCLex library__](https://pypi.org/project/NRCLex/)
     * [__RoBERTa base emotion__](https://huggingface.co/cardiffnlp/roberta-base-emotion)
   * Toxicity features
     * [__toxic BERT__](https://huggingface.co/unitary/toxic-bert) 
   * Readability features
     * __Kincaid Index__
     * __ARI Index__
     * __Coleman Index__
     * __Gunning-Fog Index__
     * __LIX__
     * __SMOG__
     * __Dale Chall__
   * Other features
     * __Number of first person pronouns (I, me or mine)__
     * __Number of words__
     * __Number of words in uppercase__
     * __Number of sentences__
     * __Number of paragraphs__
     * __Number of long words: greater or equal than 7 characters__
     * __Number of complex words__
     * __Number of interrogative and exclamation signs__
     * __Number of quantifiers: some, several, a number of, enough, numerous, plenty of, a lot of, lots of, much, many, few, little__
* __Transformers based approach__ (experiment "a posteriori" of submission deadline): on the other hand, we contrast previous approach with several transformers-based approach, using Weight and Biases.
  * [__RoBERTa-base-BNE__](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne): RoBERTa trained on text data from Biblioteca Nacional de España.
  * [__Bertin-RoBERTa-base-spanish__](https://huggingface.co/bertin-project/bertin-roberta-base-spanish?text=Fui+a+la+librer%C3%ADa+a+comprar+un+%3Cmask%3E.): RoBERTa trained from scratch on the Spanish portion of mC4 dataset.

# Early detection of depression

Once the candidate model is trained, as test data is released in rounds, inference is applied on test data
in a particular way:
  1. For each round, data is retrieved in JSON file, with same format as train data.
  2. Once it is retrieved, translation and feature engineering are applied for each message.
  3. Next, features are grouped by user id, following same format as training data.
  4. Then, we load top three models to predict labels, as well as efficiency metrics, including: __RAM needed__, __Percentage of CPU usage__, __Floating Point Operations per Second (FLOPS)__, __Total time to process (in milliseconds)__, __Kg in CO2 emissions. For this, the [Code Carbon tool is used](https://pypi.org/project/codecarbon/)__.
  5. Finally, we submit predictions for each model.

We iterate over this loop until maximum round is reached (i.e, there is no data to retrieve).
Thus, it acts as an “early detection” of depression iterator, checking in which round a user
has been diagnosed with depression.

![image](https://github.com/sjzafra/spanish-depression-detector/assets/45654081/cd2f84ad-7983-47d6-8daa-6ade53814d20)

# Results

| Model name | Accuracy          | Recall | Precision | F1 | Macro F1 | Micro F1 | ERDE 5 | ERDE 30 |
|--------------|-------------------|------------|------------|------------|------------|------------|----------|----------|
| Baseline (best model from competition) | 0.738 | 0.749 (macro) | 0.756 (macro) | ... | 0.737 | ... | 0.421 | 0.161 | 
| Bertin-RoBERTa-base-spanish | 0.5772 | 0.6736        |  0.9559        |  0.52        | 0.5368        | 0.9697        | 0.9697 | 0.802 |
| PlanTL-GOB-ES/roberta-base-bne | 0.7718 | 0.8235        |  0.7179        |  0.7671        | 0.7717        | 0.7718        | 0.9697 | 0.802 |

# References

* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
* [Weight and Biases](https://wandb.ai/site)
* [HuggingFace](https://huggingface.co/)
