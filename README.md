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

![](https://i.ibb.co/d7LVpWy/output-onlinepngtools.png)

Unfairness - Fairness metric determined by the difference between probability distributions of groups of a sensitive attribute. For example, if 10% of male instances and 5% of female instances were predicted to have heart disease, then the calculated value would be |0.9 - 0.95| + |0.1 - 0.05| = 0.10.

t-SNE Visualization - Dimensionality reduction technique for visualizing high dimensional data in 2D. In this project, t-SNE visualizations are useful for seeing differences between probability distributions of a sensitive group.

### Data and Methods
Data was gathered from CDC’s annual Behavioral Risk Factor Surveillance System (BRFSS) survey. My project’s target variables for prediction were diabetes and cardiovascular disease, and the important features included race, sex, income, and age. Fairness algorithms were employed on each of these features to find disparities between sensitive groups.
Race - split into 2 groups: White, Black/African-American       		
Sex - split into 2 groups: Male, Female
Income - split into 4 groups: Very Low, Low, Medium, High    		
Age - split into 3 groups: Young (< 30), Middle (between 30 and 50), Old (> 50)

The following preprocessing steps were taken to create the data for cardiovascular disease and diabetes training:
Dropped columns with many null values, and then dropped rows with null values.
Dropped rows with unwanted target value labels - for both diseases, only kept yes/no responses.
From the remaining columns, selected relevant columns for predicting both diabetes and cardiovascular disease, such as general health or smoking status.

#### 2 Group Fairness
Two of the four sensitive attributes processed were binary groups (male/female, white/black). This elicits a different approach from attributes with greater than two groups. This is because with two groups, one group’s can be mapped to the other in a 2D space. With more than two groups, more calculations have to be made to find the most cost effective plan in a higher dimensional space, with each group adding another dimension.

A probability matrix maps each instance of one group to the other group. For example, a matrix can map every male subject’s predicted probabilities to every female subject’s predicted probabilities by calculating the distance between the probabilities with two-norm (Fig-1). Multiple matrices were necessary to create optimal transportation plans:
 - Male/Female to All (both female and male) to create transportation plan
 - Male/Female to All Accurate - Modifies Male/Female to All to retain accuracy using the following calculation: Distance between selected instances -  log(probability of female instance getting true male) * λ, where λ is a scale factor to prevent overfitting
 - Matrix All to All - used to calculate Wasserstein Distance

An Optimal Transportation plan is a cost-efficient plan moving probabilities of one group to another. Since the target variable is binary for disease prediction, this can be understood as moving probabilities of one group to another group in a 2D space. For example, an optimal transportation plan would calculate the most cost-effective way to move dirt from dirt piles to fill holes. The library Python Optimal Transport was used, which takes in sample weights and the previously calculated probability matrix and returns the solution to the transportation problem.

There are two approaches to reduce bias using the optimal transportation plans:
1. Use a group to group transportation plan to move one group’s probabilities to the other.
2. Move both groups using two different optimal transportation plans - one for each group.

For the Correct One Group approach, several steps are taken (using Sex as example, same methods apply to Race):
1. Generate Transportation plan between male and female subjects.
2. For each row of transportation plan, randomly assign a male probability to a female probability with transportation plan as weights for randomization.
3. To combine the old and new female probability, apply an α value that merges the two probabilities with the following calculation: merged proba = (1 - α) * new proba + α * old proba. An α value of 0 will be more fair, while an α value of 1 will be more accurate. This step is important to find the best compromise between accuracy and fairness.
4. Repeat 2-3 for the opposite (female -> male with transposed T plan).
5. Measure accuracy and unfairness separately for male -> female and female -> male.

To use the Correct Both Groups approach, slightly different steps are taken:
1. Obtain Transportation plan between male and all subjects for given α and probability matrix.
2. For each row of transportation plan, randomly assign a male or female probability to a female probability with transportation plan as weights for randomization. The new female probability can either be mapped to another female instance or a male instance.
3. Repeat 2 for the opposite (male or female -> male) to generate new male probabilities.
4. Measure accuracy and unfairness with combined new probabilities for male and female.


