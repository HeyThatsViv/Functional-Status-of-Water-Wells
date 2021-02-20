'''
Functions written for use in this project to predict functional status of water wells in Tanzania
    - Vivienne DiFrancesco
    - viviennedifrancesco@gmail.com ''' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
import sklearn.metrics as metrics
from yellowbrick.classifier import ROCAUC

def fix_lat(x, df):
    '''
    Fill the zero latitude values in the data with the average latitude of
    other wells that are in the same area. Ignores entries that are not zero.
    This function must be run before longitude as it depends on the longitude
    values being zero to fill the latitude appropriately.
    
    Args:
        x (row): Datapoint of a dataframe
        df (dataframe): Dataframe that the function is referencing from
    
    Returns:
        Mean latitude: Average latitude value for non-zero valued wells in the
            same area as input datapoint
    Example: 
        df['latitude'] = df.apply(fix_lat, axis=1, args=([X_train]))
    '''
    
    # Leaves the latitude the same if latitude value is not zero    
    if x['longitude'] != 0: 
        lat = x['latitude']
        return lat 
    
    # Drill down to look for entries from the same region and lga(if possible)
    # Find the mean of the similar entries, not including those that are zero
    else:
        lga = x['lga']
        region = x['region']
        tempdf = df[(df['lga'] == lga) & (df['longitude'] !=0)]
        
        if tempdf.shape[0] != 0:
            lat_mean = tempdf['latitude'].mean()
            
        # Find similar entries by region if lga can't be matched            
        else:
            tempdf = df[(df['region'] == region) & 
                             (df['longitude'] !=0)]
            lat_mean = tempdf['latitude'].mean()
            
        return lat_mean

def fix_long(x, df):
    '''
    Fill the zero longitude values in the data with the average longitude of
    other wells that are in the same area. Ignores entries that are not zero.
    
    Args:
        x (row): Datapoint of a dataframe
        df (dataframe): Dataframe that the function is referencing from
    
    Returns:
        Mean longitude: Average longitude value for non-zero valued wells in 
            the same area as input datapoint
    Example: 
        df['longitude'] = df.apply(fix_long, axis=1, args=([X_train]))
    '''
    
    # Leaves the longitude the same if longitude value is not zero
    if x['longitude'] != 0:
        long = x['longitude']
        return long
    
    # Drill down to look for entries from the same region and lga(if possible)
    # Find the mean of the similar entries, not including those that are zero    
    else:
        lga = x['lga']
        region = x['region']
        tempdf = df[(df['lga'] == lga) & (df['longitude'] !=0)]
        
        if tempdf.shape[0] != 0:
            long_mean = tempdf['longitude'].mean()
        
        # Find similar entries by region if lga can't be matched
        else:
            tempdf = df[(df['region'] == region) & 
                             (df['longitude'] !=0)]
            long_mean = tempdf['longitude'].mean()
            
        return long_mean

def servicing_lab(x, dict_list):
    '''
    Find the assigned servicing water lab for each entry in the dataframe
    
    Args:
        x (row): Datapoint of the dataframe
        dict_list (list of dictionaries): dictionary entry for each laboratory 
        
    Returns:
        Servicing water lab assigned to the well according to location
    
    Example:
        df['water_lab'] = df.apply(servicing_lab, axis=1, args=([labs_dict_list]))
    '''
    
    reg = x['region']
    lga = x['lga']
    lab_list = []
    lga_lookup = ['Kilimanjaro', 'Manyara', 'Geita', 'Njombe', 'Tabora', 
                  'Simiyu']
    
    # Looking through the list of water lab dictionaries for the appropriate lab
    for dic in dict_list:
        
        # Matching by lga when appropriate
        if reg in lga_lookup:
            if lga in dic['LGA'].split(', '):
                lab = dic['CITY OF LABORATORY'] 
                lab_list.append(str(lab))
                
        # Matching by region when appropriate
        else:
            if dic['SERVICED REGIONS'] == reg:
                lab = dic['CITY OF LABORATORY']
                lab_list.append(lab)
    
    return lab_list[0]


def distance_servicing_lab(x, dict_list):
    '''
    Find the euclidean distance to the water lab for each entry in the 
    dataframe using latitude and longitude entries
    
    Args:
        x (row): Datapoint of the dataframe
        dict_list (list of dictionaries): dictionary entry for each laboratory
        
    Returns:
        Servicing water lab distance based on euclidean distance
    
    Example:
        df['lab_distance'] = df.apply(distance_servicing_lab, axis=1, args=([labs_dict_list]))
    '''
    
    lab = x['servicing_water_lab']
    A = [x['latitude'], x['longitude']]

    # Find the distance between the lab and the well using latitude and 
    # longitude
    for dic in dict_list:
        if lab == dic['CITY OF LABORATORY']:
            B = [float(dic['LATITUDE']), float(dic['LONGITUDE'])]
            dist = distance.euclidean(A, B)
            return dist


def closest_city(x, dict_list):
    '''
    Find the closest city to a well based on euclidean distance.
    
    Args:
        x (row): Datapoint of a dataframe
        dict_list (list of dictionaries): dictionary entry for latitude and longitude of biggest Tanzanian cities
        
    Returns:
        Closest city: Closest city to the well as a string
        
    Example:
        df['city'] = df.apply(closest_city, axis=1, args=([cities_dict_list]))
    '''

    A = [x['latitude'], x['longitude']]
    distance = 10000 # Variable to be overwritten
    city = '' # Variable to be overwritten
    
    # Looping through each city to calculate the distance
    for dic in dict_list:
        B = [float(dic['LATITUDE']), float(dic['LONGITUDE'])]
        B = np.array(B)
        dist = np.linalg.norm(A - B)
        
        # Replacing the city value if current one is closer
        if dist < distance:
            distance = dist
            city = dic['CITY']
            
    return city


def city_distance(x, dict_list):
    '''
    Find the euclidean distance between a well and the closest city.
    
    Args:
        x (row): Datapoint of a dataframe
        dict_list (list of dictionaries): dictionary entry for latitude and longitude of biggest Tanzanian cities
        
    Returns:
        Distance: Euclidean distance between well and closest city
        
    Example:
        df['distance'] = df.apply(city_distance, axis=1, args=([cities_dict_list]))
    '''
    
    city = x['closest_city']
    A = [x['latitude'], x['longitude']]

    # Using the latitude and longitude of the city and the well to get distance
    for dic in dict_list:
        if city == dic['CITY']:
            B = [float(dic['LATITUDE']), float(dic['LONGITUDE'])]
            dist = distance.euclidean(A, B)
            return dist    


def funder_assignment(x, fund_list):
    '''
    Buckets the funder values that are not in the list of most common funders 
    into an 'other' category.
    
    Args:
        x (str): funder string value in dataframe
        fund_list (list): list of most common funders in the training set
        
    Returns:
        Changes the value to 'other' if x is not in the list of most common
        funders. Otherwise leaves x unchanged.
        
    Example:
        df['funder'] = df['funder'].apply(funder_assignment, args=([fund_list]))
    '''
    
    if x not in fund_list:
        x = 'other'
    else:
        pass
    return x


def installer_assignment(x, install_list):
    '''
    Buckets the installer values that are not in the list of most common 
    installers into an 'other' category.
    
    Args:
        x (str): installer string value in dataframe
        install_list (list): list of most common funders in the training set
        
    Returns:
        Changes the value to 'other' if x is not in the list of most common
        installers. Otherwise leaves x unchanged.
        
    Example:
        df['installer'] = df['installer'].apply(installer_assignment, args=([install_list]))
    '''
    if x not in install_list:
        x = 'other'
    else:
        pass
    return x


def plotting_counts(df, col, target='status_group'):
    '''
    Generates countplot on a column in a dataframe.
    
    Args:
        df (dataframe): Dataframe that contains the column and target to be 
        plotted
        col (str): Column name of the data to be plotted against the target
        target (str): Target column of the dataframe
        
    Returns:
        Count plot figure with bars grouped by the target
    
    Example:
        plotting_counts(data, 'feature_name')
    '''

    # Sort the column values for plotting
    order_list = list(df[col].unique())
    order_list.sort()
    
    # Plot the figure
    fig, ax = plt.subplots(figsize=(16,8))
    x, y = col, target
    ax = sns.countplot(x=x, hue=y, data=df, order=order_list)

    # Set labels and title
    plt.title(f'{col.title()} By Count {target.title()}', 
              fontdict={'fontsize': 30})
    plt.xlabel(f'{col.title()}', fontdict={'fontsize': 20})
    plt.ylabel(f'{target.title()} Count', fontdict={'fontsize': 20})
    plt.xticks(rotation=75)
    return fig, ax

def plotting_percentages(df, col, target='status_group'):
    '''
    Generates catplot on a column in a dataframe that shows percentages at the
    top of each bar.
    
    Args:
        df (dataframe): Dataframe that contains the column and target to be 
        plotted
        col (str): Column name of the data to be plotted against the target
        target (str): Target column of the dataframe
        
    Returns:
        Catplot figure with bars grouped by the target and representing
        percentages of the entries for each value
    
    Example:
        plotting_percentages(data, 'feature_name')
    '''
    
    x, y = col, target
    
    # Temporary dataframe with percentage values
    temp_df = df.groupby(x)[y].value_counts(normalize=True)
    temp_df = temp_df.mul(100).rename('percent').reset_index()

    # Sort the column values for plotting    
    order_list = list(df[col].unique())
    order_list.sort()

    # Plot the figure
    sns.set(font_scale=1.5)
    g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=temp_df, 
                    height=8, aspect=2, order=order_list, legend_out=False)
    g.ax.set_ylim(0,100)

    # Loop through each bar in the graph and add the percentage value    
    for p in g.ax.patches:
        txt = str(p.get_height().round(1)) + '%'
        txt_x = p.get_x() 
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)
        
    # Set labels and title
    plt.title(f'{col.title()} By Percent {target.title()}', 
              fontdict={'fontsize': 30})
    plt.xlabel(f'{col.title()}', fontdict={'fontsize': 20})
    plt.ylabel(f'{target.title()} Percentage', fontdict={'fontsize': 20})
    plt.xticks(rotation=75)
    return g

def plot_num_cols(df, col, target='status_group'):
    '''
    Generates 'boxen' type catplot on a column in a dataframe grouped by target
    
    Args:
        df (dataframe): Dataframe that contains the column and target to be 
        plotted
        col (str): Column name of the data to be plotted against the target
        target (str): Target column of the dataframe
        
    Returns:
        Catplot 'boxen' figure split by the target 
    
    Example:
        plotting_num_cols(data, 'feature_name')
    '''
    # Generating the figure
    g = sns.catplot(x=target, y=col, data=df, kind='boxen', 
                    height=7, aspect=2)

    # Setting the title
    plt.suptitle(f'{col.title()} and {target.title()}', fontsize=30, y=1.05)


def make_classification_report(model, y_true, x_test, title=''):
    
    '''
    Generate and return the classification report for a model.
    
    Args: 
        Model (classification model): SKlearn compatable model.
        y_true (series or array): True labels to compare predictions
        x_test (dataframe or array): X data to generate predictions for
        title (str): Title for the report
        
    Returns:
        Dictionary of the classification results
    
    Example:
        make_classification_report(logreg_model, y_test, X_test, 
                                    title='Logistic Regression Model')
        
        '''
    # Generate predictions
    y_preds = model.predict(x_test)
    print('__________________________________________________________________')
    print(f'CLASSIFICATION REPORT FOR: \n\t{title}')
    print('__________________________________________________________________')
    print('\n')
    
    # Generate report
    report = metrics.classification_report(y_true, y_preds, 
                                    target_names=['functional', 'needs repair', 
                                             'nonfunctional'])
    report_dict = metrics.classification_report(y_true, y_preds, output_dict=True,
                            target_names=['functional', 'needs repair', 
                                     'nonfunctional'])
    # Add the title to the report dictionary
    report_dict['title'] = title
    print(report)
    print('__________________________________________________________________')
    
    return report_dict

def plot_confusion_matrix(model, X, y, title=''):
    '''
    Plots the normalized confusion matrix for a model
    
    Args:
        Model (classification model): SKlearn compatable model
        X (dataframe or array): feature columns of a dataframe
        y (series or array): target column of a dataframe
        title (str): Title for the matrix
    
    Returns:
        Plotted figure of the confusion matrix for the model
    
    Example:
        plot_confusion_matrix(logreg_model, X_test, y_test, 
        title='Logistic Regression Model')
    '''
    
    # Plot the matrix with labels    
    fig = metrics.plot_confusion_matrix(model, X, y, normalize='true', 
                                        cmap='Greens',
                                 display_labels=['functional', 'needs repair', 
                                             'nonfunctional'])
    # Remove grid lines
    plt.grid(False)
    
    # Set title
    plt.title(f'Confusion Matrix For {title}', fontdict={'fontsize':17})
    plt.show()
    print('__________________________________________________________________')
    return fig

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):
    '''
    Plots the ROC AUC curves for a model
    
    Args:
        Model (classification model): SKlearn compatable model
        xtrain (dataframe or array): feature columns of the training set
        ytrain (series or array): target column of the training set
        xtest (dataframe or array): feature columns of the test set
        ytest (series or array): target column of the test set
        
    Returns:
        Plotted figure of ROC AUC curves for the model
    
    Example:
        plot_ROC_curve(logreg_model, X_train, y_train, X_test, y_test)
    '''

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'functional', 
                                        1: 'needs repair', 
                                        2: 'nonfunctional'})
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    return visualizer


def plot_top_features(model, xtrain, title=''):
    '''
    Plots the top important features of a tree based model
    
    Args:
        Model (classification model): SKlearn compatable model
        xtrain (dataframe or array): feature columns for the training set
        title (str): Title for the plot
        
    Returns:
        Plotted figure of feature importances for the model
    
    Example:
        plot_top_features(rf_model, X_train, title='Random Forest Model')
    '''

    # Turn the feature importances into a series 
    importances = pd.Series(model.feature_importances_, index=xtrain.columns)
    
    # Plot the top most important features
    importances.nlargest(20).sort_values().plot(kind='barh')
    plt.title(f'Most Important Features For {title}', fontdict={'fontsize':17})
    plt.xlabel('Importance')
    return importances.sort_values(ascending=False)


def evaluate_model(model, xtrain, ytrain, xtest, ytest, 
                   tree=False, title=''):
    '''
    Runs all the evaluation functions on a model including the classification 
    report, confusion matrix, ROC AUC plot, and a top features plot if the 
    model is tree based.
    
    Args:
        model (classification model): SKlearn compatable model
        xtrain (dataframe or array): feature columns of the training set
        ytrain (series or array): target column of the training set
        xtest (dataframe or array): feature columns of the test set
        ytest (series or array): target column of the test set
        tree (boolean): if the model is tree based or not
        title (str): Title for the model
    
    Returns:
        The classification report, confusion matrix, ROC AUC plot, and top
        features plot if tree=True
    
    Example:
        evaluate_model(logreg_model, X_train, y_train, X_test, y_test,
                        title='Logistic Regression Model')
        
    '''
    
    make_classification_report(model, ytest, xtest, title=title)
    plot_confusion_matrix(model, xtest, ytest, title=title)
    plot_ROC_curve(model, xtrain, ytrain, xtest, ytest)
    
    # Feature importance can only be run on tree based models
    if tree:
        plot_top_features(model, xtrain, title=title)