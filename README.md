# Enhancing Soil Health: The Role of Machine Learning in Advancing Sustainable Farming Practices
## Case Study: Optimizing Crop Selection Through Predictive Analytics

*“The best land to plant and grow something new again is rock bottom. In that sense, hitting rock bottom, although extremely painful, is also the ground to sow new life on.”
— Dr. Clarissa Pinkola Estés, Women Who Run With the Wolves*

Farmers face a new challenge in light of climate change, albeit also an opportunity for a (farm) fresh start. According a report by the The Intergovernmental Science-Policy Platform on Biodiversity and Ecosystem Services (IPBES), an estimated 90% of the Earth’s topsoil is likely to be at risk by 2050. With more than 600 million people projected to face hunger globally by 2030, the United Nations’s goal towards Zero Hunger will require an increase in the adoption of sustainable agriculture practices to ensure adequate and nutritious food security.

An overreliance on synthetic fertilizers and pesticides has led to soil degradation, leading to a loss in soil fertility and organic matter, and an increase in soil erosion. The resulting agricultural runoff has led to increased water and land pollution, adversely impacting biodiversity and the human population at large. Improved sustainable agriculture practices that could address the current challenges include using no-till, cover cropping, and diverse crop rotations. Regenerative agriculture offers an even more promising alternative which abandons the use of synthetic fertilizers and pesticides altogether.

## Case Study: Optimizing Crop Selection Through Predictive Analytics*

Consider a common real-life scenario, where a farmer with a limited budget needs to determine the most crucial soil measure for predicting the best crop to plant on a specific plot of land. Each crop requires optimal soil conditions to achieve the best growth and highest yield.

We can build a multiclass classification model to perform feature selection to identify the soil feature with the highest predictive accuracy for classifying crop types. The dataset was obtained from DataCamp, and includes:

**Explanatory (or independent) variables, which, as the name suggests explains the response variable:**
- “N”: Nitrogen content ratio in the soil
- “P”: Phosphorous content ratio in the soil
- “K”: Potassium content ratio in the soil
- “pH”: value of the soil

**And one response (or dependent) variable, which responds to the explanatory variables:**
- “crop”: categorical values that contain various crops

Data preprocessing comes first and the one thing we immediately notice is the scale of values of pH. Phosphorous and potassium also have some outliers that can be removed.

![Screenshot 2024-08-29 225651](https://github.com/user-attachments/assets/76a1e05d-950e-4903-b013-6b0af05dcc57)

![Screenshot 2024-08-29 212458](https://github.com/user-attachments/assets/ac693c1f-fbaa-404b-9e33-5b49a89cf410)

pH has much smaller scale of values in comparison to the other features. When features are on different scales, those with larger magnitudes may receive disproportionately higher weights, potentially overshadowing features with smaller scales. To get the features to a comparable scale, we will perform some data transformations for scaling using StandardScaler from scikit-learn. StandardScaler standardizes features to have a mean of zero and a standard deviation of one.

We can now begin creating the training and test sets using all features. We will split the data into 70% train and 30% test sets.


    # Create a variable containing all features except for 'crop' column
    X = crops.drop(columns = "crop")
    
    # Create a variable containing only the 'crop' column
    y = crops["crop"]
    
    # Split into training and test sets to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)


Next, we will build a multiclass logistic regression model, typically used with dependent categorical variables, to predict the crop type using each individual feature. Therefore, we will build four models for K, N, P, and ph by looping through each feature.

    # Create an empty dictionary to store each features predictive performance
    feature_performance = {}
    
    # Loop through features
    for feature in ["N", "P", "K", "ph"]:
        # Instantiate a StandardScaler object
        scaler = StandardScaler()
        # Construct a list of steps containing tuples with step names as strings
        # Create model ensuring multi-class predicition is supported
        steps = [("scaler", StandardScaler()),
                ("log_reg", LogisticRegression(multi_class="multinomial"))]
        # Impute using a pipeline
        pipeline = Pipeline(steps)
        # Fit the model to the feature in X_train, subsetting it with brackets
        log_reg_scaled = pipeline.fit(X_train[[feature]], y_train)
        # Predict target values using the test set and storing results in y_pred
        y_pred = log_reg_scaled.predict(X_test[[feature]]) 
        
        # Calculate F1 score, the harmonic mean of precision and recall
        f1 = f1_score(y_test, y_pred, average="weighted")
        # Add F1 score pairs to the dictionary
        feature_performance[feature] = f1
        print(f"F1-score for {feature}: {f1}")
    
The output for the above code block is:
- F1-score for N: 0.07597350285728291
- F1-score for P: 0.11295396163044077
- F1-score for K: 0.13622076698553245
- F1-score for ph: 0.03842281397167993

**The highest F1 score and therefore the model with the best performance is K. However, given the generally low F1 scores, we should explore additional models and perform comparative analyses to evaluate for model outperformance.**

Accurately predicting crop types based on current nutrient levels and overall soil health can optimize crop selection, resulting in higher yields and better long-term soil maintenance. This approach supports sustainable agricultural practices, and helps farmers mitigate risks stemming from climate change risks and control disease and pest outbreaks. Ultimately, machine learning and automation are vital tools for boosting agricultural productivity and advancing food security, contributing to the United Nations’ Zero Hunger goal.
