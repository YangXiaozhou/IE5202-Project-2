def cal_cv_rmse(X, orig_data, diff_order, fold, p, d, q, log):
    """Calculate root mean square 1-step-ahead forecasting error
       based on timeseries split cross validation
       
       params:
       X: data after order differencing
       orig_data: original data (could be log-transformed)
       diff_order: order of differencing in the trans_data
       fold: cross validation fold
       p, d, q: int, params for ARIMA
       log: boolean, True is X is log-transformed data
       
       return:
       RMSE: list, list of RMSE for all folds
    """
    tscv = TimeSeriesSplit(n_splits=20)

    RMSE = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        model = ARIMA(X_train, order=(p, d, q))  
        results_ns = model.fit(disp=-1) 
        
        # forcast
        forecasts = results_ns.forecast(X_test.size)[0]

        # errors 
        errors = []
        
        # get last two values from the original space
        second_last = orig_data.loc[X_train.index][-2]
        last = orig_data.loc[X_train.index][-1]
        
        if diff_order == 1:
            # first prediction
            forecasts[0] = forecasts[0] + last
            if log:
                errors.append(np.exp(forecasts[0]) - np.exp(orig_data.loc[X_test.index][0]))
            else:
                errors.append(forecasts[0] - orig_data.loc[X_test.index][0])
            
            for i in range(diff_order, X_test.size):
                # to correct for first order differencing
                forecasts[i] = forecasts[i] + orig_data.loc[X_test.index][i-1]
                if log:
                    errors.append(np.exp(forecasts[i]) - np.exp(orig_data.loc[X_test.index][i])) 
                else:
                    errors.append(forecasts[i] - orig_data.loc[X_test.index][i])

        
        if diff_order == 2:
            # first two predictions
            pred_1 = forecasts[0] + 2*last - second_last
            pred_2 = forecasts[1] + 2*pred_1 - last
            forecasts[0] = pred_1
            forecasts[1] = pred_2
            if log:
                errors.append(np.exp(pred_1) - np.exp(orig_data.loc[X_test.index][0]))
                errors.append(np.exp(pred_2) - np.exp(orig_data.loc[X_test.index][1]))
            else:
                errors.append(pred_1 - orig_data.loc[X_test.index][0])
                errors.append(pred_2 - orig_data.loc[X_test.index][1])
            for i in range(diff_order, X_test.size):
                # to correct for second order differencing
                forecasts[i] = forecasts[i] + 2*orig_data.loc[X_test.index][i-1] - orig_data.loc[X_test.index][i-2]
                if log:
                    errors.append(np.exp(forecasts[i]) - np.exp(orig_data.loc[X_test.index][i]))    
                else:
                    errors.append(forecasts[i] - orig_data.loc[X_test.index][i])    
        
        RMSE.append(np.sqrt(np.mean(np.power(errors, 2))))              
        
    return RMSE

def cal_cv_rmse_reg(X, formula, fold):
    """Calculate root mean square forecasting error
       based on timeseries split cross validation, regression model
       on data is used.
       
       params:
       X: dataframe, data for fitting
       formula: string, formula for ols regression
       fold: int, cross validation fold
       
       return:
       RMSE: list, list of RMSE for all folds
    """ 
    
    RMSE = []
    tscv = TimeSeriesSplit(n_splits=fold)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lmfit = smf.ols(formula, data = X_train).fit()
        
        forecasts = lmfit.predict(X_test.drop('JPY_USD', 1))
        errors = forecasts - X_test['JPY_USD']
        RMSE.append(np.sqrt(np.mean(np.power(errors, 2))))
        
    return RMSE    








