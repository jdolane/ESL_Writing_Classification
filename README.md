# ESL Writing Classification
<p>The latest version of deployed model is on the <a href="https://gradespeare.streamlit.app/" target="_blank">GradeSpeare</a> app.</p>

- [Overview](#overview)
- [Datasets](#datasets)
- [Compilation and Cleaning](#compilation-and-cleaning)
- [Augmentation and Balancing](#augmentation-and-balancing)
- [NLP Dependency Matching](#nlp---dependency-matching-and-doc-vectors)
- [Model Selection](#model-selection)
- [References](#references)

## Overview
<p>This is my capstone project for the <a href="https://www.concordiabootcamps.ca/lp/data-science-lp-2?utm_term=data%20science%20course&utm_campaign=Search_MTL&utm_source=adwords&utm_medium=ppc&hsa_acc=3838886679&hsa_cam=21258988525&hsa_grp=159297764102&hsa_ad=698842094834&hsa_src=g&hsa_tgt=kwd-27111326778&hsa_kw=data%20science%20course&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQjwmMayBhDuARIsAM9HM8cIwxytsOtn7U2dY3yU9LpthFIRA6TdJUJIYlJ55XLvv580bPaAh50aAtfmEALw_wcB" target="_blank">Concordia Data Science Diploma program.</a> Its goal is to create a model that can predict the level of a written text according to the <a href="https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions#:~:text=The%20CEFR%20organises%20language%20proficiency,needs%20of%20the%20local%20context." target="_blank">CEFR</a>.</p>
<p>The current model is a multi-layer perceptron (MLP) classifier, and it predicts the level with 75% overall accuracy. It was trained on 5,943 texts from the <a href="https://github.com/ELI-Data-Mining-Group/PELIC-dataset/" target="_blank">PELIC</a> dataset, 194 texts from the <a href="https://cental.uclouvain.be/team/atack/cefr-asag/">ASAG</a> dataset, and 862 artificially augmented texts. The level that is predicted is a combination of the 'level_id' variable from the PELIC dataset, and the 'grade_majority_vote' variable from the ASAG dataset. 'level_id' indicates the level of the English course that the writer was taking at the time of production. 'grade_majority_vote' indicates the majority vote of three grades given by trained TOEFL examiners.</p>

## Datasets
<ol>
  <li>The University of Pittsburgh English Language Institute Corpus (<a href ="https://github.com/ELI-Data-Mining-Group/PELIC-dataset/" target="_blank">PELIC</a>, Juffs 2020)</li>
  <li>Université Catholique de Louvain - CEFR Automated Short Answer Grading (<a href='https://cental.uclouvain.be/team/atack/cefr-asag/' target="_blank">ASAG</a>, Tack et al.)</li>
</ol>
<p><b>PELIC</b> and <b>ASAG</b> compiled datasets are stored as .pkl files in the <a href="https://github.com/jdolane/ESL_Writing_Classification/tree/main/data" target="_blank">data</a> folder. The pickle format was chosen to conserve the datatype format of the 'doc_vector' column, which is converted to a string in .csv.</p>

## Compilation and Cleaning
<p><a href="https://github.com/jdolane/ESL_Writing_Classification/tree/main/notebooks/compile" target="_blank">Compilation</a> and <a href="https://github.com/jdolane/ESL_Writing_Classification/tree/main/notebooks/clean" target="_blank">cleaning</a> steps are demonstrated in .ipynb files in their respective subfolders in the <a href="https://github.com/jdolane/ESL_Writing_Classification/tree/main/notebooks" target="_blank">notebooks</a> folder.</p>

<p>The <b>PELIC</b> dataset consists of five .csv files that were merged with Pandas. The dataset initially consisted of 47,667 rows and 47 columns. The two main variables that were used in training the model were 'answer', which is the student's written text, and 'level_id', which is the level of the course that the student was taking at the time of writing.</p>

<p>Null values and texts of insufficient quality were removed from the dataset in such a way as to conserve as much data as possible. First, different versions of the same texts were removed, as they were essentially duplicates. Next, the answers from different course types and question types were inspected to see which questions allowed for an open text answer (many of the questions in the dataset only allowed for a selection answer, which wouldn't provide good data). It was found that all course and question types could be conserved, and only answers that were not produced in an open text field were removed.</p>

<p>After that, null values needed to be taken care of. Before dropping null values from the entire dataset, columns containing fewer than 36,304 answers were removed. These columns consisted of several variables such as birth year, gender, and test scores. It was considered that test scores could also be used as potential 'y' variables; however, keeping them in the dataset would have reduced the number of texts - it was decided that 'level_id' would serve as the data label.</p>

<p>The <b>ASAG</b> dataset consists of 299 .xml files that were scraped using <b>BeautifulSoup</b>. The questions, answers, and grades in this dataset are very clear and didn't need to be cleaned.</p>

<p>Instead of choosing a minimum answer length, <b>spaCy</b> was used to filter out answers that did not contain at least one subject and one verb. This allowed for the conservation of data from level 2, and eliminated one-word responses and multiple-choice answers. No maxiumum answer length was set.</p>

## Augmentation and Balancing
<p>The <b>PELIC</b> dataset was very imbalanced by level, with the following value counts:</p>

| level | count |
|-------|-------|
| 4     | 12163 |
| 5     | 10094 |
| 3     | 7993  |
| 2     | 849   |

<p>To address this issue, the level 2 class was doubled using GPT2Tokenizer and GPT2LMHeadModel. The texts were augmented by using AI to rephrase and generate a continuation of each answer. The second half of the AI generated texts were then truncated to get the augmented data sample. The generator uses top-k and nucleus sampling, which helps to retain the style of the text. It was considered that a simpler augmentation technique could be used, such as random shuffling and insertion, or synonym replacement; however, this wouldn't have conserved the grammatical structure of the answers, which is needed to be able to match patterns. The augmentation function is found in <a href="https://github.com/jdolane/ESL_Writing_Classification/blob/main/notebooks/augment/Augment.ipynb" target="_blank">Augment.ipynb</a></p>

<p>Once the answers of the level 2 class were augmented, the answers from the remaining classes (3, 4, and 5) were reduced. The reduction was not random; rather, a function was used to choose the longest answers first, and to not choose an answer from the same question twice, where possible, until the data was balanced. The balancing function is found in the notebooks in the <a href="https://github.com/jdolane/ESL_Writing_Classification/tree/main/notebooks/balance" target="_blank">balancing</a> folder.</p>

<img src="images/augmented_by_dataset.png" alt="Count of Augmented Rows by Dataset" width="50%">

## NLP - Dependency Matching and Doc Vectors
<p>Patterns for 26 verb tense combinations, 3 gerund dependencies, and two modal verbs were defined using spaCy's DependencyMatcher. These patterns, along with the number of sentences and the average sentence length, </p>

## Model Selection
## References

- Juffs, A., Han, N-R., & Naismith, B. (2020). The University of Pittsburgh English Language Corpus (PELIC) [Data set]. <a href="http://doi.org/10.5281/zenodo.3991977" target="_blank">http://doi.org/10.5281/zenodo.3991977</a>
- Tack, Anaïs, Thomas François, Sophie Roekhaut, and Cédric Fairon. (2017) "Human and Automated CEFR-based Grading of Short Answers." In Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications, pp. 169-179. Association for Computational Linguistics, 2017. [Paper](https://www.aclweb.org/anthology/W17-5018) [DOI](https://doi.org/10.18653/v1/W17-5018)

