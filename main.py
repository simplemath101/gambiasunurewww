"""
main.py
Main entry point for AI-powered trading bot
Integrates all components: data, model, strategy, broker, and backtesting
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import argparse

from data_handler import DataHandler
from model import TradingModel
from strategy import TradingStrategy, RiskManager
from broker import Broker
from backtest import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main AI-powered trading bot that integrates all components
    """
    
    def __init__(self, config: dict):
        """
        Initialize trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.data_handler = DataHandler(
            exchange_name=config['exchange'],
            symbol=config['symbol']
        )
        
        self.model = TradingModel(
            model_type=config['model_type']
        )
        
        self.strategy = TradingStrategy(
            rsi_oversold=config['rsi_oversold'],
            rsi_overbought=config['rsi_overbought'],
            ml_confidence_threshold=config['ml_confidence_threshold'],
            use_ml=config['use_ml']
        )
        
        self.risk_manager = RiskManager(
            max_position_size=config['max_position_size'],
            max_daily_loss=config['max_daily_loss'],
            max_trades_per_day=config['max_trades_per_day'],
            min_risk_reward=config['min_risk_reward']
        )
        
        self.broker = Broker(
            exchange_name=config['exchange'],
            paper_trading=config['paper_trading']
        )
        
        self.is_trained = False
        self.running = False
        
        logger.info("Trading bot initialized")
    
    def train_model(self, 
                   historical_periods: int = 1000,
                   train_split: float = 0.8) -> dict:
        """
        Train ML model on historical data
        
        Args:
            historical_periods: Number of historical periods to fetch
            train_split: Train/test split ratio
            
        Returns:
            Training and evaluation metrics
        """
        logger.info("Starting model training...")
        
        try:
            # Fetch historical data
            df = self.data_handler.fetch_ohlcv(
                timeframe=self.config['timeframe'],
                limit=historical_periods
            )
        except Exception as e:
            logger.warning(f"Could not fetch real data: {e}. Using synthetic data.")
            df = self.data_handler.generate_sample_data(periods=historical_periods)
        
        # Add indicators and features
        df = self.data_handler.add_technical_indicators(df)
        df = self.data_handler.create_features(df, lookback=10)
        df = self.data_handler.create_labels(df, forward_periods=1)
        
        # Prepare data for ML
        ml_data = self.data_handler.prepare_ml_data(df, train_split=train_split)
        
        # Train model
        train_metrics = self.model.train(
            ml_data['X_train'],
            ml_data['y_train'],
            ml_data['feature_names']
        )
        
        # Evaluate model
        eval_metrics = self.model.evaluate(
            ml_data['X_test'],
            ml_data['y_test']
        )
        
        self.is_trained = True
        
        # Save model
        self.model.save_model()
        
        logger.info(f"Model training completed. "
                   f"Train Accuracy: {train_metrics['train_accuracy']:.4f}, "
                   f"Test Accuracy: {eval_metrics['test_accuracy']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'ml_data': ml_data
        }
    
    def backtest_strategy(self, 
                         historical_periods: int = 1000,
                         use_ml: bool = True) -> dict:
        """
        Backtest trading strategy on historical data
        
        Args:
            historical_periods: Number of historical periods
            use_ml: Whether to use ML predictions
            
        Returns:
            Backtest results
        """
        logger.info("Starting backtest...")
        
        # Fetch and prepare data
        try:
            df = self.data_handler.fetch_ohlcv(
                timeframe=self.config['timeframe'],
                limit=historical_periods
            )
        except:
            df = self.data_handler.generate_sample_data(periods=historical_periods)
        
        df = self.data_handler.add_technical_indicators(df)
        df = self.data_handler.create_features(df, lookback=10)
        
        # Generate signals
        signals = []
        for i in range(len(df)):
            if i < 50:  # Skip first 50 periods for indicator warmup
                signals.append('HOLD')
                continue
            
            row_data = df.iloc[i]
            
            # Get ML prediction if enabled and model is trained
            ml_prediction = None
            ml_confidence = None
            
            if use_ml and self.is_trained:
                try:
                    # Prepare features
                    feature_cols = [col for col in df.columns 
                                  if col not in ['label', 'label_binary', 'future_return',
                                               'open', 'high', 'low', 'close', 'volume']]
                    features = row_data[feature_cols].values.reshape(1, -1)
                    
                    # Get prediction
                    pred, conf = self.model.predict(features)
                    ml_prediction = 'BUY' if pred == 1 else 'SELL'
                    ml_confidence = conf
                except Exception as e:
                    logger.debug(f"Could not get ML prediction: {e}")
            
            # Generate signal
            signal_data = self.strategy.generate_signal(
                row_data,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence
            )
            
            signals.append(signal_data['signal'])
        
        signals_series = pd.Series(signals, index=df.index)
        
        # Run backtest
        backtester = Backtester(
            initial_capital=self.config['initial_capital'],
            commission=self.config['commission']
        )
        
        results = backtester.run_backtest(
            df,
            signals_series,
            stop_loss_pct=self.config['stop_loss_pct'],
            take_profit_pct=self.config['take_profit_pct']
        )
        
        # Print and plot results
        backtester.print_summary(results)
        backtester.plot_results(results, save_path='backtest_results.png')
        
        return results
    
    def live_trading_loop(self, update_interval: int = 60):
        """
        Main live trading loop
        
        Args:
            update_interval: Time between updates in seconds
        """
        logger.info("Starting live trading loop...")
        self.running = True
        
        last_signal = None
        
        while self.running:
            try:
                # Get latest market data
                df = self.data_handler.get_latest_data(
                    timeframe=self.config['timeframe']
                )
                
                if df.empty:
                    logger.warning("No data received, skipping iteration")
                    time.sleep(update_interval)
                    continue
                
                current_data = df.iloc[-1]
                current_price = current_data['close']
                
                # Get ML prediction
                ml_prediction = None
                ml_confidence = None
                
                if self.config['use_ml'] and self.is_trained:
                    try:
                        feature_cols = [col for col in df.columns 
                                      if col not in ['label', 'label_binary', 'future_return',
                                                   'open', 'high', 'low', 'close', 'volume']]
                        features = current_data[feature_cols].values.reshape(1, -1)
                        pred, conf = self.model.predict(features)
                        ml_prediction = 'BUY' if pred == 1 else 'SELL'
                        ml_confidence = conf
                    except Exception as e:
                        logger.error(f"ML prediction error: {e}")
                
                # Generate trading signal
                signal_data = self.strategy.generate_signal(
                    current_data,
                    ml_prediction=ml_prediction,
                    ml_confidence=ml_confidence
                )
                
                signal = signal_data['signal']
                
                # Check if we should enter a trade
                if signal != 'HOLD' and signal != last_signal:
                    should_trade = self.strategy.should_enter_trade(signal_data)
                    
                    if should_trade:
                        self._execute_signal(signal, current_data)
                
                last_signal = signal
                
                # Log status
                position = self.broker.get_position(self.config['symbol'])
                balance = self.broker.get_balance()
                portfolio_value = self.broker.get_portfolio_value([self.config['symbol']])
                
                logger.info(f"Status - Price: ${current_price:.2f}, "
                          f"Signal: {signal}, "
                          f"Position: {position['size']:.6f}, "
                          f"Balance: ${balance:.2f}, "
                          f"Portfolio: ${portfolio_value:.2f}")
                
                # Wait before next iteration
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                logger.info("Received stop signal")
                self.stop()
                break
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(update_interval)
    
    def _execute_signal(self, signal: str, current_data: pd.Series):
        """
        Execute trading signal
        
        Args:
            signal: Trading signal
            current_data: Current market data
        """
        symbol = self.config['symbol']
        current_price = current_data['close']
        balance = self.broker.get_balance()
        position = self.broker.get_position(symbol)
        
        if signal == 'BUY' and position['size'] == 0:
            # Calculate position size
            position_size = self.strategy.calculate_position_size(
                account_balance=balance,
                risk_per_trade=self.config['risk_per_trade'],
                current_price=current_price,
                stop_loss_pct=self.config['stop_loss_pct']
            )
            
            if position_size > 0:
                # Calculate stop loss and take profit
                atr = current_data.get('atr', None)
                stop_loss = self.strategy.calculate_stop_loss(
                    entry_price=current_price,
                    signal='BUY',
                    atr=atr,
                    stop_loss_pct=self.config['stop_loss_pct']
                )
                take_profit = self.strategy.calculate_take_profit(
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    signal='BUY',
                    risk_reward_ratio=self.config['risk_reward_ratio']
                )
                
                # Validate trade
                is_valid, reason = self.risk_manager.validate_trade(
                    position_size=position_size,
                    account_balance=balance,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if is_valid:
                    # Place market order
                    order = self.broker.place_market_order(
                        symbol=symbol,
                        side='buy',
                        amount=position_size
                    )
                    
                    if order:
                        logger.info(f"BUY order executed: {position_size:.6f} {symbol} @ ${current_price:.2f}")
                        
                        # Place stop loss order
                        self.broker.place_stop_loss_order(
                            symbol=symbol,
                            side='sell',
                            amount=position_size,
                            stop_price=stop_loss
                        )
                else:
                    logger.warning(f"Trade rejected: {reason}")
        
        elif signal == 'SELL' and position['size'] > 0:
            # Close position
            order = self.broker.place_market_order(
                symbol=symbol,
                side='sell',
                amount=position['size']
            )
            
            if order:
                # Calculate P&L
                pnl = (current_price - position['entry_price']) * position['size']
                self.risk_manager.update_trade_metrics(pnl)
                
                logger.info(f"SELL order executed: {position['size']:.6f} {symbol} @ ${current_price:.2f}, "
                          f"P&L: ${pnl:.2f}")
    
    def stop(self):
        """Stop trading bot"""
        logger.info("Stopping trading bot...")
        self.running = False
        
        # Close any open positions
        position = self.broker.get_position(self.config['symbol'])
        if position['size'] > 0:
            logger.info("Closing open position...")
            self.broker.place_market_order(
                symbol=self.config['symbol'],
                side='sell',
                amount=position['size']
            )
        
        logger.info("Trading bot stopped")


def load_config(config_path: str = 'config.json') -> dict:
    """Load configuration from file"""
    default_config = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'model_type': 'random_forest',
        'paper_trading': True,
        'initial_capital': 10000,
        'commission': 0.001,
        
        # Strategy parameters
        'use_ml': True,
        'ml_confidence_threshold': 0.65,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        
        # Risk management
        'risk_per_trade': 0.02,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'risk_reward_ratio': 2.0,
        'max_position_size': 0.1,
        'max_daily_loss': 0.05,
        'max_trades_per_day': 10,
        'min_risk_reward': 1.5,
    }
    
    # Try to load from file
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Created default configuration file: {config_path}")
    
    return default_config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live'],
                       default='backtest', help='Operating mode')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize bot
    bot = TradingBot(config)
    
    if args.mode == 'train':
        logger.info("=== TRAINING MODE ===")
        results = bot.train_model(historical_periods=1000, train_split=0.8)
        print("\n=== Training Results ===")
        print(f"Train Accuracy: {results['train_metrics']['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['eval_metrics']['test_accuracy']:.4f}")
        print("\nTop 5 Important Features:")
        for feat, imp in list(results['train_metrics']['feature_importance'].items())[:5]:
            print(f"  {feat}: {imp:.4f}")
    
    elif args.mode == 'backtest':
        logger.info("=== BACKTEST MODE ===")
        
        # Train model first if using ML
        if config['use_ml']:
            logger.info("Training model for backtest...")
            bot.train_model(historical_periods=1000)
        
        # Run backtest
        results = bot.backtest_strategy(
            historical_periods=1000,
            use_ml=config['use_ml']
        )
    
    elif args.mode == 'live':
        logger.info("=== LIVE TRADING MODE ===")
        
        if not config['paper_trading']:
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Exiting...")
                return
        
        # Train or load model
        if config['use_ml']:
            try:
                bot.model.load_model()
                bot.is_trained = True
                logger.info("Loaded existing model")
            except:
                logger.info("No existing model found, training new model...")
                bot.train_model(historical_periods=1000)
        
        # Start live trading
        bot.live_trading_loop(update_interval=60)


if __name__ == "__main__":
    main()
