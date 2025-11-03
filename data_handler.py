"""
data_handler.py
Handles fetching and preprocessing market data from exchanges
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Fetches and preprocesses market data from cryptocurrency exchanges
    Supports multiple exchanges via CCXT library
    """
    
    def __init__(self, exchange_name: str = 'binance', symbol: str = 'BTC/USDT'):
        """
        Initialize data handler
        
        Args:
            exchange_name: Name of exchange (binance, alpaca, etc.)
            symbol: Trading pair symbol
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logger.info(f"Connected to {self.exchange_name}")
            return exchange
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
    
    def fetch_ohlcv(self, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data
        
        Args:
            timeframe: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                timeframe=timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        
        # ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Trend strength (ADX)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        logger.info("Technical indicators added")
        return df
    
    def create_features(self, df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Create feature engineering for ML model
        
        Args:
            df: DataFrame with technical indicators
            lookback: Number of periods to look back for features
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Lagged features
        for i in range(1, lookback + 1):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # Price position relative to moving averages
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # Volatility features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        logger.info("Feature engineering completed")
        return df
    
    def create_labels(self, df: pd.DataFrame, forward_periods: int = 1, 
                     threshold: float = 0.002) -> pd.DataFrame:
        """
        Create target labels for ML model
        
        Args:
            df: DataFrame with features
            forward_periods: Number of periods to look ahead
            threshold: Minimum return to classify as buy/sell
            
        Returns:
            DataFrame with labels added
        """
        df = df.copy()
        
        # Future returns
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Classification labels: 0=hold, 1=buy, 2=sell
        df['label'] = 0  # Hold
        df.loc[df['future_return'] > threshold, 'label'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'label'] = 2  # Sell
        
        # Binary classification (simplified)
        df['label_binary'] = (df['future_return'] > 0).astype(int)
        
        logger.info("Labels created")
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame, 
                       train_split: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Prepare data for ML model training
        
        Args:
            df: DataFrame with features and labels
            train_split: Ratio of training data
            
        Returns:
            Dictionary containing train/test splits
        """
        # Remove NaN values
        df_clean = df.dropna()
        
        # Select feature columns (exclude target and timestamp-related columns)
        feature_cols = [col for col in df_clean.columns 
                       if col not in ['label', 'label_binary', 'future_return', 
                                     'open', 'high', 'low', 'close', 'volume']]
        
        X = df_clean[feature_cols].values
        y = df_clean['label_binary'].values
        
        # Train-test split (time series - no shuffle)
        split_idx = int(len(X) * train_split)
        
        data = {
            'X_train': X[:split_idx],
            'X_test': X[split_idx:],
            'y_train': y[:split_idx],
            'y_test': y[split_idx:],
            'feature_names': feature_cols,
            'df_test': df_clean.iloc[split_idx:]
        }
        
        logger.info(f"Data prepared: {len(data['X_train'])} training samples, "
                   f"{len(data['X_test'])} test samples")
        
        return data
    
    def get_latest_data(self, timeframe: str = '1h') -> pd.DataFrame:
        """
        Get latest processed data for live trading
        
        Args:
            timeframe: Candlestick timeframe
            
        Returns:
            Latest data point with all features
        """
        df = self.fetch_ohlcv(timeframe=timeframe, limit=100)
        df = self.add_technical_indicators(df)
        df = self.create_features(df)
        
        return df.iloc[-1:]
    
    def generate_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data for testing (when exchange is unavailable)
        
        Args:
            periods: Number of periods to generate
            
        Returns:
            Synthetic OHLCV DataFrame
        """
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate synthetic price data with trend and noise
        np.random.seed(42)
        trend = np.linspace(40000, 45000, periods)
        noise = np.random.randn(periods) * 500
        close_prices = trend + noise
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(periods) * 100,
            'high': close_prices + abs(np.random.randn(periods) * 150),
            'low': close_prices - abs(np.random.randn(periods) * 150),
            'close': close_prices,
            'volume': np.random.randint(100, 1000, periods)
        }, index=dates)
        
        logger.info(f"Generated {periods} periods of synthetic data")
        return df


if __name__ == "__main__":
    # Test the data handler
    handler = DataHandler(exchange_name='binance', symbol='BTC/USDT')
    
    try:
        # Try to fetch real data
        df = handler.fetch_ohlcv(timeframe='1h', limit=500)
    except:
        # Fall back to synthetic data
        logger.warning("Using synthetic data for testing")
        df = handler.generate_sample_data(periods=500)
    
    # Add indicators and features
    df = handler.add_technical_indicators(df)
    df = handler.create_features(df)
    df = handler.create_labels(df)
    
    print("\nData shape:", df.shape)
    print("\nFeature columns:", df.columns.tolist())
    print("\nLast 5 rows:")
    print(df[['close', 'rsi', 'macd', 'label_binary']].tail())
    
    # Prepare for ML
    ml_data = handler.prepare_ml_data(df)
    print(f"\nML data prepared:")
    print(f"Training samples: {len(ml_data['X_train'])}")
    print(f"Test samples: {len(ml_data['X_test'])}")
    print(f"Number of features: {len(ml_data['feature_names'])}")
