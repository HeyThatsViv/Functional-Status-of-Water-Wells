# Predicting Functional Status Of Water Wells In Tanzania

**Author**: Vivienne DiFrancesco

The contents of this repository detail an analysis of classification of Tanzanian water wells as either functional, non functional, or needs repair. This analysis is detailed in hopes of making the work accessible and replicable.

## Repository Structure

- README.md: The top level README for reviewers of this project
- main_notebook.ipynb: narritive documentation of analysis in jupyter notebook
- TanzaniaWaterWellsSlides.pdf: pdf version of project presentation slides
- Data folder: Contains datasets used in this project

![MainImage](https://raw.githubusercontent.com/HeyThatsViv/Functional-Status-of-Water-Wells/master/Visuals/MainImage.jpg)

## Business problem

The purpose of this project is to use machine learning classification models to predict the functional status of water wells in Tanzania. The different status groups for classification are functional, non functional, and functional but needs repair. The hope is that by predicting the functional status of a well, access to water could be improved across Tanzania.

## Data
The data used for this project is from the Data Driven website where, at the time of completing this project, it is an active competition. The dataset contains nearly 60,000 records of water wells across Tanzania. Each record has information that includes various location data, technical specifications of the well, information about the water, etc. The website provides a list of the features contained and a brief description of each. The link to the website to obtain the data for yourself is: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

You can also get the .csv files from the "Data" folder of this repository.


## Methods
The approach for this project was to create many different model types to see what performs the best and to compare and contrast the different types of models. The way the data was preprocessed with feature engineering, filling missing values, and scaling was done with the goal of increasing accuracy of the models. The OSEMiN process is the overarching structure of this project. 

For each type of model, a model was first trained and fitted with default parameters as a base. Then, key parameters were chosen to tune using sklearn GridSearchCV and the best parameters were used to run the model. Finally, the tuned parameters were used to fit the same model using the dataset after SMOTE had been performed. This was partially a practical choice, as grid search tuning takes a lot of time and SMOTE add extra rows to the data which would add extra time. But another purpose was to evaluate the difference in performance with SMOTE data versus using the class weight parameter. Performance was compared to the base model of each type, as well as between different model types.

## Results

### Random Forest Confusion Matrix
![confusionmatrix](https://raw.githubusercontent.com/AnyOldRandomNameWillDOo/Module-3-Final-Project/master/Visuals/ConfustionMatrixRandomForest.png)
> Confusion matrix results of the random forest model using SMOTE data

### Water Well Status By Location
![location](https://raw.githubusercontent.com/AnyOldRandomNameWillDOo/Module-3-Final-Project/master/Visuals/WaterWellStatusByLocation.png)
> The location of the water well is a high predictor of functional status.


## Recommendations

- It is difficult to say what is the "best" model because there is a trade-off with the models having higher accuracy versus the ability to predict the needs repair class. Many of the more complicated models were not any better than simpler models in accuracy or at predicting the wells that need repair. The SMOTE version of random forest seems like the best middle ground for accuracy, computational simplicity, and having any hope of predicting the needs repair class. It was also the model that categorized the non functional wells as being functional the least.

- There is a dramatic difference in the construction year and the functional status of the well with older wells being more likely to be non functional or need repair. With further analysis an ideal time frame of when to service wells could be found that would balance cost and prevention.

- Wells that are known to be dry and older wells should be considered for looking at most closely as that would make a well most likely to be non functional.

- The features that were added of distance to the nearest city and servicing water lab came up often as important features so they should be added to the data collection process of the wells.

- Ultimately to get better results, more features will need to be added during the data collection process. Including information such as when the well was last serviced, what kind of repairs have been done on the wells, if any parts have been replaced, etc. These examples also seem that they would be very useful in giving the models more meaning for the needs repair class.

- Having so many of the models use location data like latitude, longitude, region, and LGA as important features for prediction means that where the well is located is a big factor of whether it is working or not. This matter should be investigated to try to bridge any disparity gaps and bring more reliable water wells to the regions with the most non functioning wells.

## Further Directions

- Preprocessing with clusters - Using something like the KMeans clustering package from sklearn, adding cluster features to the data may be a way to add more separation in the data between the classes. It could even be interesting to find if certain models work better on the different clusters.

- Catboost - Catboost is a modeling package that is supposed to work well on data with many categorical features. This dataset has more categorical features than numerical so it may be worth testing.

- Further tuning or more model types - It's possible that more fine tuning of models or continuing to try different model types could bring more accuracy. There was not time to do more testing with the stacked models and the meta classifiers which is a prime example of something to try.

- More feature engineering - More research could possibly bring up more features to add to the dataset like were found with the servicing water labs. Or if there was some manipulating of certain features in the dataset to bring out more meaningful features.




