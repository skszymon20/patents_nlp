
# U.S. Patent Phrase to Phrase Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)

The task is to train the model on a novel semantic similarity dataset
to extract relevant information by matching key phrases in patent documents.
Determining the semantic similarity between phrases is critically important during the patent search
and examination process to determine if an invention has been described before.
For example, if one invention claims "television set" and a prior publication describes "TV set",
a model would ideally recognize these are the same and assist a patent attorney or examiner in retrieving relevant documents.
This extends beyond paraphrase identification; if one invention claims a "strong material" and another uses "steel",
that may also be a match. What counts as a "strong material" varies per domain (it may be steel in one domain
and ripstop fabric in another, but you wouldn't want your parachute made of steel).
We have included the Cooperative Patent Classification as the technical domain context as an additional feature
to help you disambiguate these situations.

### Data description

The dataset has pairs of phrases (an anchor and a target phrase) and the task is to rate how similar they are on a scale 
from 0 (not at all similar) to 1 (identical in meaning). This challenge differs from a standard semantic similarity task
in that similarity has been scored here within a patent's context, specifically its [CPC classification](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification)
(version 2021.05), which indicates the subject to which 
the patent relates. For example, while the phrases "bird" and "Cape Cod" may have low semantic similarity in normal language,
the likeness of their meaning is much closer if considered in the context of "house".
Information on the meaning of CPC codes may be found on the [USPTO website](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html).

#### Train set

Contains phrases, contexts, and their similarity scores.

Columns meaning:
- id - a unique identifier for a pair of phrases
- anchor - the first phrase
- target - the second phrase
- context - the CPC classification (version 2021.05), which indicates the subject within which the similarity is to be scored
- score - the similarity. This is sourced from a combination of one or more manual expert ratings.

Head of the table:

| id               | anchor    | target                 | context | score |
|:----------------:|:---------:|:----------------------:|:-------:|:-----:|
| 37d61fd2272659b1 | abatement | abatement of pollution | A47     | 0.5   |
| 7b9652b17b68b7a4 | abatement | act of abating         | A47     | 0.75  |
| 36d72442aefd8232 | abatement | active catalyst        | A47     | 0.25  |

#### Test set

Identical in structure to the training set but without the score.

Head of the table:

| id               | anchor          | target                        | context |
|:----------------:|:---------------:|:-----------------------------:|:-------:|
| 4112d61851461f60 | opc drum        | inorganic photoconductor drum | G02     |
| 09e418c93a776564 | adjust gas flow | altering gas flow             | F23     |
| 36baf228038e314b | lower trunnion  | lower locating                | B60     |

Data for this competition is derived from the public archives of the U.S. Patent and Trademark Office (USPTO).
These archives, offered in machine-readable formats, have already enabled numerous AI research projects such as
the training of large-scale language models. Beyond enabling AI research, patent documents can also be used to understand
the dynamics of AI innovation at large.

### Scoring

The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- 1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
- 0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- 0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- 0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- 0.0 - Unrelated.

Results are evaluated on the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
between the predicted and actual similarity score.

The whole description and additional information on the task can be found [here](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching).
