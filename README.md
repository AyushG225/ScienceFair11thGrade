# ScienceFair11thGrade
Minimizing Discriminatory Bias in Healthcare Models Through Optimal Transportation and Threshold-Agnostic Fairness

### Introduction
As artificial intelligence becomes more prevalent in making critical decisions within healthcare, concerns have emerged regarding biases within these AI models, which can lead to unfair predictions and disparities in diagnoses among different groups. 
Various approaches can be employed to reduce these disparities, including preprocessing to eliminate sensitive information such as sex or race, static threshold approaches where fairness is measured based on whether the predictive score surpasses a predefined threshold, and post-processing to rectify biases within the model. However, with preprocessing, there is a risk that the algorithm may inadvertently infer biases from other features in the data. 

For example, if certain socioeconomic factors are closely associated with race or sex in the dataset, the algorithm may still make decisions based on these factors even if explicit demographic information is excluded. In addition, static threshold approaches may neglect the dynamic nature of the real world, because it does not consider changes in data distributions. 

Therefore, this project aims to leverage threshold-agnostic post-processing algorithms and various fairness metrics to mitigate discriminatory predictions against specific groups of people.

Engineering Goal: The objectives of this project are to:
1. Create healthcare machine learning models to predict disease susceptibility using sensitive attributes.
2. Determine bias within models through various fairness metrics.
3. Optimally reduce bias within models through post-processing algorithms while maintaining model performance.

Project Origin + Previous Work: I was originally exposed to discrimination in machine learning from my science fair project last year, which analyzed the impacts of Social Determinants of Health on local communities’ disease susceptibility. From this project, I learned that many machine learning models that are created with sensitive attributes create biases that reflect societal biases, such as racism (Obermeyer et al., 2019).

Work by Others: This project was supported by my mentor Dr. Qihang Lin, from University of Iowa. My mentor guided me to develop efficient algorithms that could be used to reduce and measure bias, and I implemented the code for this project, allowing us to iterate through new methods. 

### Definitions
Sensitive Attribute - Characteristic of an individual such as sex or race that could lead to discrimination in machine learning. “2 Groups” means that the sensitive attribute has two categories (e.g. sex - male/female), while “More than 2 Groups” means that the sensitive attribute has greater than 2 categories (e.g. age - young/middle/old).

Optimal Transportation Plan - Mathematical framework that finds the most efficient way to align probability distributions of sensitive attributes, such as race or sex. By minimizing the “cost” to transform these distributions, the plan aims to retain model performance while achieving equitable outcomes between groups. For example, the plan would attempt to move the probability distribution of female instances to male instances in a way that retains the original performance of the model.

Threshold-Agnostic Fairness - Fairness approach where fairness metrics are defined based on the distributions between demographic groups/sensitive attributes. For example, a bank could adjust its threshold based on current applicants’ credit scores and demographic groups when choosing to give a loan, to ensure fairness across groups.

Post-Processing - When adjustments are made to a model’s predictions. In the context of ML fairness, adjustments are made to reduce bias within the model’s predictions.

Wasserstein Distance - Fairness metric that measures the minimum amount of work needed to transform one probability distribution to another. Can be used to quantify the disparity between probability distributions of sensitive attributes.

Wasserstein Barycenter - Mathematical concept that uses optimal transportation to compute the central distribution given a set of distributions and minimizing transportation costs. This is useful for my project when there are more than 2 sensitive groups, which would lead to a set of probability distributions.

Calculation for Probability Matrix - Matrix norm between probability distributions of 2 groups. For example, if a male instance had a 10% predicted probability of having heart disease, while a female instance had a 5% predicted probability, then the calculated mapping value between the two instances would be:

![hello](https://i.ibb.co/kJgDd3J/image.png)

Unfairness - Fairness metric determined by the difference between probability distributions of groups of a sensitive attribute. For example, if 10% of male instances and 5% of female instances were predicted to have heart disease, then the calculated value would be |0.9 - 0.95| + |0.1 - 0.05| = 0.10.

t-SNE Visualization - Dimensionality reduction technique for visualizing high dimensional data in 2D. In this project, t-SNE visualizations are useful for seeing differences between probability distributions of a sensitive group.

