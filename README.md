# Sequential prediction model of clinical phenotypes


## Introduction
Medical concepts are inherently ambiguous, making them difficult for use in making prognoses. Our work focuses on building prediction models on top of semantic representation of concepts organized by Word2Vec. First, we transform Electronic Health Records (EHRs) into sequences of medical concepts, where each medical concept is analogous to a word in a sentence. Second, we feed the sequences into Word2Vec, which builds a vector representation for each concept. Finally, we generate various predictive models for early prognosis using this vector representation.


## Installation

This project assumes that you already have access to some MIMICIII database. If not you could follow the following instructions:

https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/

Installation steps:

    sudo apt-get install python-psycopg2
    sudo apt-get install libpq-dev
    pip install -r requirements.txt


## Run

### 1. Data Preparation
To take the code for a spin run the following commands:

    cd DataPrep/mimic
    psql -U mimic -a -f allevents.sql
    python generate_icd_levels.py
    python generate_seq_combined.py

After executing the last command you will see collection of text files populated in your Data/mimic\_seq folder. Each line in these files represent 1 patient. Here is an example:

> d_401,d_486|{"black": 0, "hispanic": 0, "other": 0, "age": 69.90554414784394, "mideast": 0, "multi": 0, "gender": 0, "hawaiian": 0, "portuguese": 0, "american": 0, "asian": 0, "white": 1}|p_ASA81 p_CALCG2gi100NS p_CALCG2100NS p_CALCG2100NS p_HEPA5I l_50970 l_51265 p_ACET325 p_HYDR2 p_VANC1F p_VANCOBASE p_HEPA10SYR p_HEPA10SYR p_METO25 l_50862 l_50954 p_POTA20 l_50924 l_50953 l_50998 d_038 d_285.9 d_401 d_486 d_584 d_995|p_ASA325 p_D545NS1000 p_DEX50SY p_DOCU100 p_DOCU100L l_51214 d_401 d_486

A line is divided by pipes "|" into 4 parts:
1. Diagnoses that we used as labels. In this example it is:

> d_401,d_486

2. Demographics in a json object. You can use this library to load them into python object:

> {"black": 0, "hispanic": 0, "other": 0, "age": 69.90554414784394, "mideast": 0, "multi": 0, "gender": 0, "hawaiian": 0, "portuguese": 0, "american": 0, "asian": 0, "white": 1}

3. Previous admission events (which includes abnormal lab tests prefixed with "l\_", prescriptions prefixed with "p\_" and diagnosis with "d\_").

> p_ASA81 p_CALCG2/100NS p_CALCG2100NS p_CALCG2100NS p_HEPA5I l_50970 l_51265 p_ACET325 p_HYDR2 p_VANC1F p_VANCOBASE p_HEPA10SYR p_HEPA10SYR p_METO25 l_50862 l_50954 p_POTA20 l_50924 l_50953 l_50998 d_038 d_285.9 d_401 d_486 d_584 d_995

4. The final admission which we used in our project as a hold out set. You would also notice that the diagnoses in part 1 also exists in this part:

> p_ASA325 p_D545NS1000 p_DEX50SY p_DOCU100 p_DOCU100L l_51214 d_401 d_486


### 2. Prediction
To run Patient Diagnosis Projection Similarity (PDPS) run the following commands:

    cd ../../Prediction
    python pdps.py --dataset mimic --window 30 --size 350 --decay 8

Patient Diagnosis Concept Similarity:

    python pdcs.py --dataset mimic --window 30 --size 350 --decay 8

Collaborative Filtering:

    python collaborative.py --dataset mimic --window 30 --size 350 --decay 8

Temporal tf-idf:

    python ttfidf.py --dataset mimic --ngrams 2 --skip 10 --decay 8 --prior 1

Note that Ttfidf and Collaborative filtering approaches are computationally expensive and will require considerable amount of time to execute. All prediction methods execute 10-fold cross vaildation and outputs the following files:
* A csv file containing various metrics (AUC, Accuracy, F-Score, Sensitivity, Specificity, TP, FP, TN, FN) for each diagnosis. This file is created in `Results/Stats` directory.

* ROC plot stored in `Results/Plots`:

![ROC curve](https://github.com/wael34218/MimicVectorPredictor/blob/master/Results/Plots/ROC_SkipGram_ba=False_da=mimic_de=10.0_mo=org_pr=False_si=350_wi=23.png?raw=true){:width="400px"}


Vector representations constructed by Word2Vec are able to capture semantic meaning of medical concepts. Word2Vec clusters concepts based on their type as shown in the figure. In addition, it was able to capture closely similar concepts, for example the cosine similarity of ’p_WARF2’ (Warfarin 2mg Tab) and ’p_WARF1’ (Warfarin 1mg Tab) is 0.924. All prescriptions starting with ’p_WARF’  are close to each other around the point (0, 0.45). This representation simplifies learning since it does not treat similar concepts as different features.

![Word Vectors](https://github.com/wael34218/MimicVectorPredictor/blob/master/Results/Analysis/word_vectors.png?raw=true){:width="400px"}


## Libraries Used

This project depends on:

1. word2vec - https://code.google.com/p/word2vec/
2. ICD9 - https://github.com/sirrice/icd9
3. ICD9 - https://github.com/kshedden/icd9
4. gensim - http://radimrehurek.com/gensim/
