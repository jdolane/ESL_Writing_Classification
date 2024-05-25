# ESL Writing Classification
<p>The latest version of deployed model is on the <a href="https://gradespeare.streamlit.app/" target="_blank">GradeSpeare</a> app.</p>

## Overview
<p>This is my capstone project for the <a href="https://www.concordiabootcamps.ca/lp/data-science-lp-2?utm_term=data%20science%20course&utm_campaign=Search_MTL&utm_source=adwords&utm_medium=ppc&hsa_acc=3838886679&hsa_cam=21258988525&hsa_grp=159297764102&hsa_ad=698842094834&hsa_src=g&hsa_tgt=kwd-27111326778&hsa_kw=data%20science%20course&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQjwmMayBhDuARIsAM9HM8cIwxytsOtn7U2dY3yU9LpthFIRA6TdJUJIYlJ55XLvv580bPaAh50aAtfmEALw_wcB" target="_blank">Concordia Data Science Diploma program.</a> Its goal is to create a model that can predict the level of a written text according to the <a href="https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions#:~:text=The%20CEFR%20organises%20language%20proficiency,needs%20of%20the%20local%20context." target="_blank">CEFR</a>.</p>
<p>The current model is a multi-layer perceptron (MLP) classifier, and it predicts the level with 75% overall accuracy. It was trained on 5,943 texts from the <a href="https://github.com/ELI-Data-Mining-Group/PELIC-dataset/" target="_blank">PELIC</a> dataset, 194 texts from the <a href="https://cental.uclouvain.be/team/atack/cefr-asag/">ASAG</a> dataset, and 862 artificially augmented texts. The level that is predicted is a combination of the 'level_id' variable from the PELIC dataset, and the 'grade_majority_vote' variable from the ASAG dataset. 'level_id' indicates the level of the English course that the writer was taking at the time of production. 'grade_majority_vote' indicates the majority vote of three grades given by trained TOEFL examiners.</p>

## Datasets
<ol>
  <li>The University of Pittsburgh English Language Institute Corpus (<a href ="https://github.com/ELI-Data-Mining-Group/PELIC-dataset/" target="_blank">PELIC</a>)</li>
  <li>Université Catholique de Louvain - CEFR Automated Short Answer Grading (<a href='https://cental.uclouvain.be/team/atack/cefr-asag/'>ASAG</a>, Tack, Anaïs et al.)</li>
  <li>[ASAG](#asag)</li>
</ol>

## Compilation
## Cleaning
## Augmentation
## Balancing
## NLP Dependency Matching
## Model Selection
## References

- <a name="asag"></a>Tack, Anaïs, Thomas François, Sophie Roekhaut, and Cédric Fairon. "Human and Automated CEFR-based Grading of Short Answers." In Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications, pp. 169-179. Association for Computational Linguistics, 2017. [Paper](https://www.aclweb.org/anthology/W17-5018) [DOI](https://doi.org/10.18653/v1/W17-5018)
