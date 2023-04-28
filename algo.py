from AlgorithmImports import *
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm
import math


class XGP(QCAlgorithm):
    def Initialize(self):
        # set up 
        self.SetStartDate(2023, 4, 19)
        self.SetEndDate(2023, 4, 26)

        self.InitCash = 10000000
        self.SetCash(self.InitCash)
        self.lookback = 25

        # setting brokerage and reality modeling params 
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(lambda s : s.SetSlippageModel(VolumeShareSlippageModel()))

        # manually keeping track of securities 
        self.securities = []

        # We're getting an error when this is removed
        self.weights = []
        self.trained = True

        self.risk_threshold = -0.015
        # self.SetWarmup(timedelta(days=50))
        self.prices_at_order = {}
        self.yesterday_total_profit = 0
        self.yesterday_total_fees = 0
        
        # Requesting data
        self.AddUniverseSelection(FineFundamentalUniverseSelectionModel(self.SelectCoarse, self.SelectFine))
        self.UniverseSettings.Resolution = Resolution.Daily
        self.num_coarse_symbols = 100
        self.num_fine_symbols = 5

        # Train immediately
        self.Train(self.DateRules.On(2023, 4, 20), self.TimeRules.At(4,0), self.classifier_training)
        self.Schedule.On(self.DateRules.On(2023, 4, 20), self.TimeRules.At(10,0), self.actions)

        # Train every Sunday at 4am or first day of month (because new universe)
        self.Train(self.DateRules.Every(DayOfWeek.Sunday), self.TimeRules.At(4, 0), self.classifier_training)
        self.Train(self.DateRules.MonthStart(daysOffset = 0), self.TimeRules.At(4, 0), self.classifier_training)

        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday),
                        self.TimeRules.At(10, 0),
                        self.actions)

        self.Schedule.On(self.DateRules.MonthStart(daysOffset = 0),
                        self.TimeRules.At(10, 0),
                        self.actions)

    def SelectCoarse(self, coarse):
        """selecting CoarseFundamental objects based on criteria in paper"""
        if self.Time.day == 20 and self.Time.month == 4:
            selected = [c for c in coarse if c.HasFundamentalData and c.Price > 5]
            sorted_by_dollar_volume = sorted(selected, key=lambda c: c.DollarVolume, reverse=True)
            return [c.Symbol for c in sorted_by_dollar_volume[:self.num_coarse_symbols]]    
        if self.Time.day != 1:
            return Universe.Unchanged
        selected = [c for c in coarse if c.HasFundamentalData and c.Price > 5]
        sorted_by_dollar_volume = sorted(selected, key=lambda c: c.DollarVolume, reverse=True)
        return [c.Symbol for c in sorted_by_dollar_volume[:self.num_coarse_symbols]]

    def SelectFine(self, fine):
        """selecting FineFundamental objects based on our criteria"""
        if self.Time.day == 20 and self.Time.month == 4:
            selected = [f for f in fine if 
                        f.ValuationRatios.PERatio < 100 
                        and f.MarketCap > 300000000
                        and f.ValuationRatios.PEGRatio < 3
                        and f.OperationRatios.TotalDebtEquityRatio.Value < 2
                        and f.OperationRatios.CurrentRatio.Value > 1]
            sorted_by_pe_ratio = sorted(selected, key=lambda f: f.ValuationRatios.PERatio, reverse=True)
            return [f.Symbol for f in sorted_by_pe_ratio[:self.num_fine_symbols]]            
        if self.Time.day != 1:
            return Universe.Unchanged
        selected = [f for f in fine if 
                    f.ValuationRatios.PERatio < 100 
                    and f.MarketCap > 300000000
                    and f.ValuationRatios.PEGRatio < 3
                    and f.OperationRatios.TotalDebtEquityRatio.Value < 2
                    and f.OperationRatios.CurrentRatio.Value > 1]
        sorted_by_pe_ratio = sorted(selected, key=lambda f: f.ValuationRatios.PERatio, reverse=True)
        return [f.Symbol for f in sorted_by_pe_ratio[:self.num_fine_symbols]]

    def OnSecuritiesChanged(self, changes):
        """triggers when Universe changes as result of filtering"""

        for security in changes.AddedSecurities:
            self.Debug(f"{self.Time}: Added {security}")

        for security in changes.RemovedSecurities:
            self.Debug(f"{self.Time}: Removed {security}")

        added = changes.AddedSecurities
        removed = changes.RemovedSecurities
        self.securities = list(set(self.securities).union(set(added)).difference(set(removed)))

    def OnData(self, data):
        return
        
    def OnEndOfDay(self):  
        self.yesterday_total_profit = self.Portfolio.TotalProfit
        self.yesterday_total_fees = self.Portfolio.TotalFees

    def OnOrderEvent(self, orderEvent):
        """logs the details of an order"""
        self.Log(f'{orderEvent}')

# =====================================================================================================================
# begin custom functions
# =====================================================================================================================

    def actions(self):

        self.Liquidate() #Liquidate the whole portfolio
        self.make_predictions()
        self.LinOptProg()

        lookback = 30
        active_securities = [s.Symbol for s in self.securities]


        # risk management
        if len(self.weights) > 0:
            history = self.History(active_securities, timedelta(days=lookback), resolution=Resolution.Daily)
            history = history['close'].unstack(level=0)
            history.columns = active_securities
            returns = history.pct_change()

            w = np.array([i[0] for i in self.weights])
            VaR = self.value_at_risk(returns, w) # calculation of value-at-risk limit
            self.Debug(f"VaR={VaR}")
            
            # position sizing
            max_loss_dollars = self.InitCash * self.risk_threshold # maximum loss in dollars we are willing to have in one trade

            # self.Debug(f"max_loss_dollars={max_loss_dollars}")

            if VaR <= self.risk_threshold:  # if estimated loss in the next day is greater than our maximum risk threshold
                self.Debug(f"estimated risk {VaR} exceeds threshold")
                reduction_size = self.risk_threshold - VaR
                for (security, wt) in zip(active_securities, [i[0] for i in self.weights]):

                    quantity = self.CalculateOrderQuantity(security, wt)
                    reduced_quantity = math.ceil(quantity * 0.6)
                    if reduced_quantity != 0:
                        self.Debug(f"VaR limit reached; expected loss is {VaR}. Reducing  position size of \
                            {security} from {quantity} to {reduced_quantity}")
                        self.MarketOrder(security, reduced_quantity) 
            else:
                a_securities = [s for s in self.securities]

                for (security, wt) in zip(a_securities, [i[0] for i in self.weights]):
                    if wt != 0:
                        self.SetHoldings(security.Symbol, wt)
                        self.prices_at_order[security.Symbol] = self.Securities[security.Symbol].Price

        
    def classifier_training(self):

        self.return_mods = []
        self.quantile_mods_lg = []
        self.quantile_mods_st = []
 
        #active_securities = [s.Symbol.Value for s in self.securities]
        active_securities = [s.Symbol for s in self.securities]
        self.Log(f"Training Started at {self.Time}")
        for security in active_securities:
            data = self.get_all_data([security], training=True, backtesting=False) # get tickers
            try:
                y_reg = data["return"]
                X = data.drop(["direction", "return", "symbol"], axis = 1)
                (ret, qut_lg, qut_st) = self.gb_returns(X, y_reg)
            except:
                ret = "NoModel"
                qut_lg = "NoModel"
                qut_st = "NoModel"
            self.return_mods.append(ret)
            self.quantile_mods_lg.append(qut_lg)
            self.quantile_mods_st.append(qut_st)
        
        self.trained = True

    def make_predictions(self):

        self.returns = []
        self.quantiles = []

        act_securities = [s.Symbol for s in self.securities]
        for i in range(len(act_securities)):
            security = act_securities[i]
            data = self.get_all_data([security], training=False)
            data = data[data.index == data.index.max()]
            prediction_data = data.drop(["direction", "return", "symbol"], axis = 1)
            try:
                r_pred = self.return_mods[i].predict(prediction_data)[0]
            except:
                r_pred = 0
            
            if r_pred > 0:
                q_pred = self.quantile_mods_lg[i].predict(prediction_data)[0]
            elif r_pred < 0:
                q_pred = self.quantile_mods_st[i].predict(prediction_data)[0]
            else:
                q_pred = 0

            
            self.returns.append(r_pred)
            self.quantiles.append(q_pred)
        
        self.Debug(self.returns)


    def gb_returns(self, X, y):
        """ 
        Function to calculate expected returns and quantile loss
        """
        mean_clf = GradientBoostingRegressor(n_estimators = 150,
                                            loss = "squared_error",
                                            criterion = "friedman_mse",
                                            learning_rate = 0.05,
                                            random_state = 1693,
                                            n_iter_no_change = 15)
        mean_fit_out = mean_clf.fit(X,y)
        
        quantile_clf_lg = GradientBoostingRegressor(n_estimators = 150,
                                                    loss = "quantile",
                                                    alpha = 0.05,
                                                    n_iter_no_change = 15,
                                                    learning_rate = 0.05,
                                                    criterion = "friedman_mse",
                                                    random_state = 1693)
        quantile_clf_st = GradientBoostingRegressor(n_estimators = 150,
                                                    loss = "quantile",
                                                    alpha = 0.95,
                                                    n_iter_no_change = 15,
                                                    learning_rate = 0.05,
                                                    criterion = "friedman_mse",
                                                    random_state = 1693)

        quantile_fit_lg = quantile_clf_lg.fit(X,y)
        quantile_fit_st = quantile_clf_st.fit(X,y)
        return (mean_fit_out, quantile_fit_lg, quantile_fit_st)

    def LinOptProg(self):
        """
        Convex optimization Function
        """

        self.weights = []
        self.returns = np.array(self.returns).reshape(-1,1)
        self.quantiles = np.array(self.quantiles).reshape(-1,1)

        dirs = np.array([1 if d > 0 else 0 if d == 0 else -1 for d in self.returns]).reshape(-1,1)
        bounds = [(0, min(0.6, 3 / len(self.returns))) if d == 1 else (max(-0.6, -1.5 / len(self.returns)), 0) for d in dirs]
        A = np.array([-1*self.quantiles, dirs, -1*dirs]).squeeze()
        b = np.array([0.01, 1, 0])
        res = linprog(-1*self.returns, A_ub = A, b_ub = b, bounds = bounds)
        if res.status == 0:
            self.weights = res.x.reshape(-1,1)
        else:
            self.Log("Optimization failed")

            # If optimization fails, give uniform weight 0 (buy nothing)
            self.weights = dirs * (1/len(self.returns))
        
        del self.returns
        del self.quantiles

    def bollinger_bands(self, data, window=20, num_std=2):

        # Calculate the moving average
        data['MA'] = data['close'].rolling(window=window).mean()
        
        # Calculate the standard deviation
        data['STD'] = data['close'].rolling(window=window).std()
        
        # Calculate the Bollinger Bands
        data['Upper_BB'] = data['MA'] + (data['STD'] * num_std)
        data['Lower_BB'] = data['MA'] - (data['STD'] * num_std)
        
        return data

    def calculate_rsi(self,data, period=20):

        # Calculate the daily price changes (gains and losses)
        delta = data['close'].diff().dropna()

        # Separate gains and losses into their own series
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate the average gain and average loss
        avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate the Relative Strength Index (RSI)
        rsi = 100 - (100 / (1 + rs))
        data['rsi']=rsi

        return data

    def get_all_data(self, tickers, historical=True, training=False, backtesting=True):
        """
        Gets historical data for training and prediction

        Parameters:
        -----------
        tickers : list 
            list of tickers to retrieve data 
        historical : Bool, default True
            Flag to determine if we are training or backtesting; False if live trading 
        training : Bool, default False 
            If True, retrieves training data, a 90-day period. If performing predictions, 
            False retrieves most recent day of data. For example, if called at 8 A.M., retrieves 
            the previous trading days' data. 
        backtesting : Bool, default True
            Flag to determine if we are backtesting or training 
        
        Return:
        -------
        self.dat : pd.DataFrame 
            DataFrame containing data 
        """
        if historical: 
            if backtesting: 
                shift_factor = 30 # overshooting and select the maximum 
                hist_lookback = 1 + shift_factor
                tiingo_lookback = 12 # in case of weekends
            elif training: 
                hist_lookback = self.lookback + 25
                tiingo_lookback = self.lookback * 1.5
            else:
                raise ValueError("Please train or backtest if using historical = True")
        else:
            shift_factor = 7 # needed so we can calculate lagged data 
            hist_lookback = 1 + shift_factor
            tiingo_lookback = 1 + shift_factor # in case of weekends

        full_data = pd.DataFrame()

        for symbol in tickers:

            # Get Price History
            history = self.History(symbol, hist_lookback)
            history = pd.DataFrame(history)

            # convert the historical data to a pandas DataFrame
            history['direction'] = np.where(history['close'] > history['open'], 1, 0)
            history['return']=history['close'].pct_change(periods=5)
            history = self.bollinger_bands(history)
            history = self.calculate_rsi(history)

            # Add relevant columns
            history['price_diff']=history["open"]-history["MA"]
            history['band_diff_up']=history["open"]-history["Upper_BB"]
            history['band_diff_lower']=history["open"]-history["Lower_BB"]

            # Add Tiingo Data
            data = self.AddData(TiingoNews, symbol).Symbol
            tiingo = self.History(data, int(tiingo_lookback), Resolution.Daily)
            if len(tiingo)!=0 and set(['description','publisheddate']).issubset(tiingo.columns):
                analyzer = SentimentIntensityAnalyzer()
                tiingo['polarity'] = tiingo['description'].dropna().apply(lambda x: analyzer.polarity_scores(x))
                tiingo = pd.concat([tiingo.drop(['polarity'], axis=1), tiingo['polarity'].apply(pd.Series)], axis=1)
                tiingo = tiingo[[ 'publisheddate', 'compound']]
                tiingo['publisheddate'] = pd.to_datetime(tiingo['publisheddate'],utc=True).dt.date
                tiingo = tiingo.groupby(by=[ 'publisheddate'], as_index=False).sum()
                tiingo.rename(columns={'publisheddate' : 'time'}, inplace=True)
                tiingo.set_index('time',inplace=True)
                history = history.join(tiingo)
            
            lags = range(1,5)
            history=history.assign(**{
                 f'{col} (t-{lag})': history[col].shift(lag)
                 for lag in lags
                 for col in history
            }).dropna().drop(columns = ['close','high','low','volume'], errors='ignore')

            history['symbol'] = symbol.Value

            full_data=pd.concat([full_data, history])
        return full_data

    def get_daily_realized_pnl(self):
        daily_gross_profit = self.Portfolio.TotalProfit - self.yesterday_total_profit
        daily_fees = self.Portfolio.TotalFees - self.yesterday_total_fees
        return daily_gross_profit - daily_fees

    def value_at_risk(self, returns, weights, conf_level=0.05, num_days=1):
        """
        Calculates the value-at-risk of the portfolio.
        ---------------------------------------------------
        Parameters
        returns : pd.DataFrame
            periodic returns
        conf_level : float
            confidence level. 0.05 by default
        weights : np.array
            portfolio weights
        num_days : int
            length of the period the VaR is calculated over
        """

        cov_matrix = returns.cov()
        avg_return = returns.mean()
        portfolio_mean = avg_return.dot(weights)
        portfolio_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        cutoff = norm.ppf(conf_level, portfolio_mean, portfolio_stdev)

        # n-Day VaR
        VaR = cutoff * np.sqrt(num_days)

        return VaR

    def cvar(self, returns, weights, conf_level=0.05):
        """
        Calculates the portfolio CVaR
        ------------------------------------
        Parameters
        returns : pd.DataFrame
            portfolio returns
        stdev : 
            portfolio standard deviation
        conf_level : float
            confidence level
        """
        VaR = value_at_risk(returns, weights)
        return VaR.mean()
