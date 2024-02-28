# Naive Bayes Classifier
- very efficient supervised learning algorithm.
- scales well with high dimensions.
- not as popular right now, cause deel learning methods outperforms these
- works well even if feature dataset is small.
- Its called naive because the assumption is the features are independent of each others. Strong `Independence assumptions`
    - Ex: Feature, for detection its apple or not, feature ,color is red, diamtere is 8 cm, rounded
    - Here it considers each of this features contribute independently to predicting if the given fruit is apple or not.

# Formulae
-  P(Ck | X1 X2 … Xn) C is possible output of k classes, this is probability. X are features here.
-   P(Ck | X1 X2 … Xn) = (P(Ck) P(x | Ck)) / p(x) , P(x) is same for all so we ignore this
-   P(X1 X2 … Xn | Ck |) = P(X1|Ck) ... P(Xn|CCk)  // Probability of each feature multiplied

# Example for geeks for geeks
Copied from [GeeksForGeeks](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

![ P(A|B) = \frac{P(B|A) P(A)}{P(B)}              ](https://quicklatex.com/cache3/f9/ql_1aed84a77b63869b8c6db45dba209df9_l3.svg "Rendered by QuickLaTeX.com")

where A and B are events and P(B) ≠ 0

*   Basically, we are trying to find probability of event A, given the event B is true. Event B is also termed as ****evidence****.
*   P(A) is the ****priori**** of A (the prior probability, i.e. Probability of event before evidence is seen). The evidence is an attribute value of an unknown instance(here, it is event B).
*   P(B) is Marginal Probability: Probability of Evidence.
*   P(A|B) is a posteriori probability of B, i.e. probability of event after evidence is seen.
*   P(B|A) is Likelihood probability i.e the likelihood that a hypothesis will come true based on the evidence.

Now, with regards to our dataset, we can apply Bayes’ theorem in following way:

![ P(y|X) = \frac{P(X|y) P(y)}{P(X)}              ](https://quicklatex.com/cache3/d8/ql_07fe26be84d36837c49c3328ec5e1dd8_l3.svg "Rendered by QuickLaTeX.com")

where, y is class variable and X is a dependent feature vector (of size **n**) where:

![ X = (x\_1,x\_2,x\_3,.....,x\_n)              ](https://quicklatex.com/cache3/ba/ql_03c3b6dfff061310d40f5e599dd828ba_l3.svg "Rendered by QuickLaTeX.com")

Just to clear, an example of a feature vector and corresponding class variable can be: (refer 1st row of dataset)

    X = (Rainy, Hot, High, False)
    y = No

So basically, ![P(y|X)         ](https://quicklatex.com/cache3/2b/ql_a7ad7a7a3c243cc1cbd126abf9d1af2b_l3.svg "Rendered by QuickLaTeX.com") here means, the probability of “Not playing golf” given that the weather conditions are “Rainy outlook”, “Temperature is hot”, “high humidity” and “no wind”.

With relation to our dataset, this concept can be understood as:

*   We assume that no pair of features are dependent. For example, the temperature being ‘Hot’ has nothing to do with the humidity or the outlook being ‘Rainy’ has no effect on the winds. Hence, the features are assumed to be ****independent****.
*   Secondly, each feature is given the same weight(or importance). For example, knowing only temperature and humidity alone can’t predict the outcome accurately. None of the attributes is irrelevant and assumed to be contributing ****equally**** to the outcome.

Now, its time to put a naive assumption to the Bayes’ theorem, which is, ****independence**** among the features. So now, we split ****evidence**** into the independent parts.

Now, if any two events A and B are independent, then,

    P(A,B) = P(A)P(B)

Hence, we reach to the result:

![ P(y|x\_1,...,x\_n) = \frac{ P(x\_1|y)P(x\_2|y)...P(x\_n|y)P(y)}{P(x\_1)P(x\_2)...P(x\_n)}          ](https://quicklatex.com/cache3/62/ql_1b2dfccb5b8b232047d0b950953b8262_l3.svg "Rendered by QuickLaTeX.com")

which can be expressed as:

![ P(y|x\_1,...,x\_n) = \frac{P(y)\prod\_{i=1}^{n}P(x\_i|y)}{P(x\_1)P(x\_2)...P(x\_n)}          ](https://quicklatex.com/cache3/0a/ql_69e9586fcc286d18b2a61caa8689010a_l3.svg "Rendered by QuickLaTeX.com")

Now, as the denominator remains constant for a given input, we can remove that term:

![ P(y|x\_1,...,x\_n)\propto P(y)\prod\_{i=1}^{n}P(x\_i|y)          ](https://quicklatex.com/cache3/fe/ql_1984ba956d8625b3852c6f33e7bde9fe_l3.svg "Rendered by QuickLaTeX.com")

Now, we need to create a classifier model. For this, we find the probability of given set of inputs for all possible values of the class variable **y** and pick up the output with maximum probability. This can be expressed mathematically as:

![y = argmax\_{y} P(y)\prod\_{i=1}^{n}P(x\_i|y)          ](https://quicklatex.com/cache3/1f/ql_a213dd83d2bd6d27cfb4c52c447ee01f_l3.svg "Rendered by QuickLaTeX.com")

So, finally, we are left with the task of calculating ![ P(y)          ](https://quicklatex.com/cache3/5e/ql_09ba9edf26d6794057a0a17282402c5e_l3.svg "Rendered by QuickLaTeX.com")and ![P(x\_i | y)         ](https://quicklatex.com/cache3/a3/ql_e8d4058ad00d26908d14f73a972c12a3_l3.svg "Rendered by QuickLaTeX.com").

Please note that ![P(y)         ](https://quicklatex.com/cache3/92/ql_cbde8659e426594879ee8a16736fe692_l3.svg "Rendered by QuickLaTeX.com") is also called class probability and ![P(x\_i | y)         ](https://quicklatex.com/cache3/a3/ql_e8d4058ad00d26908d14f73a972c12a3_l3.svg "Rendered by QuickLaTeX.com") is called conditional probability.

The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of ![P(x\_i | y).         ](https://quicklatex.com/cache3/62/ql_06d52151f57f6a36095eedaeca4f4762_l3.svg "Rendered by QuickLaTeX.com")

Let us try to apply the above formula manually on our weather dataset. For this, we need to do some precomputations on our dataset.

We need to find![ P(x\_i | y\_j)          ](https://quicklatex.com/cache3/4e/ql_1e8e860e67e5b20e2b4266f9ddab4d4e_l3.svg "Rendered by QuickLaTeX.com")for each ![x\_i         ](https://quicklatex.com/cache3/be/ql_58d21f221ebca469112732c52da732be_l3.svg "Rendered by QuickLaTeX.com") in X and![y\_j         ](https://quicklatex.com/cache3/69/ql_3c4074bbac38098710c70b9994b54f69_l3.svg "Rendered by QuickLaTeX.com") in y. All these calculations have been demonstrated in the tables below:

![naive-bayes-classification](https://media.geeksforgeeks.org/wp-content/uploads/20231220123018/naive-bayes-classification.webp)

So, in the figure above, we have calculated ![P(x\_i | y\_j)         ](https://quicklatex.com/cache3/16/ql_f7b005e937f9b98267efd72ff0ce4516_l3.svg "Rendered by QuickLaTeX.com") for each ![x\_i         ](https://quicklatex.com/cache3/be/ql_58d21f221ebca469112732c52da732be_l3.svg "Rendered by QuickLaTeX.com") in X and ![y\_j         ](https://quicklatex.com/cache3/69/ql_3c4074bbac38098710c70b9994b54f69_l3.svg "Rendered by QuickLaTeX.com") in y manually in the tables 1-4. For example, probability of playing golf given that the temperature is cool, i.e P(temp. = cool | play golf = Yes) = 3/9.

Also, we need to find class probabilities ![P(y)         ](https://quicklatex.com/cache3/92/ql_cbde8659e426594879ee8a16736fe692_l3.svg "Rendered by QuickLaTeX.com") which has been calculated in the table 5. For example, P(play golf = Yes) = 9/14.

So now, we are done with our pre-computations and the classifier is ready!

Let us test it on a new set of features (let us call it today):

    today = (Sunny, Hot, Normal, False)

![ P(Yes | today) = \frac{P(Sunny Outlook|Yes)P(Hot Temperature|Yes)P(Normal Humidity|Yes)P(No Wind|Yes)P(Yes)}{P(today)}              ](https://quicklatex.com/cache3/37/ql_dc7791b83801e0158aa412d584fce337_l3.svg "Rendered by QuickLaTeX.com")

and probability to not play golf is given by:

![ P(No | today) = \frac{P(Sunny Outlook|No)P(Hot Temperature|No)P(Normal Humidity|No)P(No Wind|No)P(No)}{P(today)}              ](https://quicklatex.com/cache3/94/ql_b53e6b866ac806b5e33841367d44d694_l3.svg "Rendered by QuickLaTeX.com")

Since, P(today) is common in both probabilities, we can ignore P(today) and find proportional probabilities as:

![ P(Yes | today) \propto \frac{3}{9}.\frac{2}{9}.\frac{6}{9}.\frac{6}{9}.\frac{9}{14} \approx 0.02116              ](https://quicklatex.com/cache3/54/ql_813352fa7152dcc55ba22f0415d91c54_l3.svg "Rendered by QuickLaTeX.com")

and

![ P(No | today) \propto \frac{3}{5}.\frac{2}{5}.\frac{1}{5}.\frac{2}{5}.\frac{5}{14} \approx 0.0068              ](https://quicklatex.com/cache3/f7/ql_a052b2bc2ec44a7a5b69055a8cdcf9f7_l3.svg "Rendered by QuickLaTeX.com")

Now, since

![ P(Yes | today) + P(No | today) = 1              ](https://quicklatex.com/cache3/39/ql_4f207b6bfede1f57ae7e6dc1cd0e0639_l3.svg "Rendered by QuickLaTeX.com")

These numbers can be converted into a probability by making the sum equal to 1 (normalization):

![ P(Yes | today) = \frac{0.02116}{0.02116 + 0.0068} \approx 0.0237              ](https://quicklatex.com/cache3/1c/ql_f4621a80d5a5eb3695aea1a11f67591c_l3.svg "Rendered by QuickLaTeX.com")

and

![ P(No | today) = \frac{0.0068}{0.0141 + 0.0068} \approx 0.33              ](https://quicklatex.com/cache3/e7/ql_b1df3ec666d964e6c37129de28ce92e7_l3.svg "Rendered by QuickLaTeX.com")

Since

![ P(Yes | today) > P(No | today)              ](https://quicklatex.com/cache3/74/ql_27d48d50f46c661fee32162431693d74_l3.svg "Rendered by QuickLaTeX.com")

So, prediction that golf would be played is ‘Yes’.

The method that we discussed above is applicable for discrete data. In case of continuous data, we need to make some assumptions regarding the distribution of values of each feature. The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of ![P(x\_i | y).         ](https://quicklatex.com/cache3/62/ql_06d52151f57f6a36095eedaeca4f4762_l3.svg "Rendered by QuickLaTeX.com")
