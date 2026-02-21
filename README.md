# Awesome AI in Finance [![Awesome](https://awesome.re/badge.svg)](https://github.com/sindresorhus/awesome) ⭐ 438,982 | 🐛 70 | 📅 2026-01-28 [![Community](https://img.shields.io/discord/733027681184251937.svg?style=flat\&label=Join%20Community\&color=7289DA)](https://discord.gg/cqaUf47) with stars

There are millions of trades made in the global financial market every day. Data grows very quickly and people are hard to understand.
With the power of the latest artificial intelligence research, people analyze & trade automatically and intelligently. This list contains the research, tools and code that people use to beat the market.

\[[中文资源](origin/chinese.md)]

## Contents

* [Agents](#agents)
* [LLMs](#llms)
* [Papers](#papers)
* [Courses & Books](#courses--books)
* [Strategies & Research](#strategies--research)
  * [Time Series Data](#time-series-data)
  * [Portfolio Management](#portfolio-management)
  * [High Frequency Trading](#high-frequency-trading)
  * [Event Drive](#event-drive)
  * [Crypto Currencies Strategies](#crypto-currencies-strategies)
  * [Technical Analysis](#technical-analysis)
  * [Lottery & Gamble](#lottery--gamble)
  * [Arbitrage](#arbitrage)
* [Data Sources](#data-sources)
* [Research Tools](#research-tools)
* [Trading System](#trading-system)
* [TA Lib](#ta-lib)
* [Exchange API](#exchange-api)
* [Articles](#articles)
* [Others](#others)

## Agents

* [TradingAgents](https://github.com/TauricResearch/TradingAgents) ⭐ 30,255 | 🐛 235 | 🌐 Python | 📅 2026-02-07 - Multi-Agents LLM Financial Trading Framework.
* 🌟🌟 [nofx](https://github.com/NoFxAiOS/nofx) ⭐ 10,526 | 🐛 479 | 🌐 Go | 📅 2026-02-20 - A multi-exchange Al trading platform with multi-Ai competition self-evolution, and real-time dashboard.
* 🌟 [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) ⭐ 6,238 | 🐛 61 | 🌐 Jupyter Notebook | 📅 2026-01-30 - An Open-Source AI Agent Platform for Financial Analysis using LLMs.

## LLMs

* 🌟 [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund) ⭐ 45,819 | 🐛 52 | 🌐 Python | 📅 2026-02-02 - Explore the use of AI to make trading decisions.
* [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) ⭐ 18,627 | 🐛 83 | 🌐 Jupyter Notebook | 📅 2026-02-11 - Provides a playground for all people interested in LLMs and NLP in Finance.
* [Hands-on LLMs: Train and Deploy a Real-time Financial Advisor](https://github.com/iusztinpaul/hands-on-llms) ⚠️ Archived - Train and deploy a real-time financial advisor chatbot with Falcon 7B and CometLLM.
* 🌟🌟 [MarS](https://github.com/microsoft/MarS) ⭐ 1,644 | 🐛 10 | 🌐 Python | 📅 2025-08-06 - A Financial Market Simulation Engine Powered by Generative Foundation Model.
* [PIXIU](https://github.com/chancefocus/PIXIU) ⭐ 830 | 🐛 10 | 🌐 Jupyter Notebook | 📅 2025-03-04 - An open-source resource providing a financial large language model, a dataset with 136K instruction samples, and a comprehensive evaluation benchmark.
* 🌟🌟🌟 [Nof1](https://thenof1.com/) - Benchmark designed to measure AI's investing abilities. Each model is given $10,000 of real money, in real markets, with identical prompts and input data.
* 🌟🌟 [Financial Statement Analysis with Large Language Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835311) - GPT-4 can outperform professional financial analysts in predicting future earnings changes, generating useful narrative insights, and resulting in superior trading strategies with higher Sharpe ratios and alphas, thereby suggesting a potential central role for LLMs in financial decision-making.
* [MACD + RSI + ADX Strategy (ChatGPT-powered) by TradeSmart](https://www.tradingview.com/script/GxkUyJKW-MACD-RSI-ADX-Strategy-ChatGPT-powered-by-TradeSmart/) - Asked ChatGPT on which indicators are the most popular for trading. We used all of the recommendations given.
* [A ChatGPT trading algorithm delivered 500% returns in stock market. My breakdown on what this means for hedge funds and retail investors](https://www.reddit.com/r/ChatGPT/comments/13duech/a_chatgpt_trading_algorithm_delivered_500_returns/)
* [Use chatgpt to adjust strategy parameters](https://twitter.com/0xUnicorn/status/1663413848593031170)
* [ChatGPT Strategy by OctoBot](https://blog.octobot.online/trading-using-chat-gpt) - Use ChatGPT to determine which cryptocurrency to trade based on technical indicators.

## Papers

* [The Theory of Speculation L. Bachelier, 1900](http://www.radio.goldseek.com/bachelier-thesis-theory-of-speculation-en.pdf) - The influences which determine the movements of the Stock Exchange are.
* [Brownian Motion in the Stock Market Osborne, 1959](http://m.e-m-h.org/Osbo59.pdf) - The common-stock prices can be regarded as an ensemble of decisions in statistical equilibrium.
* [An Investigation into the Use of Reinforcement Learning Techniques within the Algorithmic Trading Domain, 2015](http://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/j.cumming.pdf)
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059.pdf)
* [Reinforcement Learning for Trading, 1994](http://papers.nips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
* [Dragon-Kings, Black Swans and the Prediction of Crises Didier Sornette](https://arxiv.org/pdf/0907.4290.pdf) - The power laws in the distributions of event sizes under a broad range of conditions in a large variety of systems.
* [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787.pdf) - Deep reinforcement learning provides a framework toward end-to-end training of such trading agent.
* [Machine Learning for Trading](https://cims.nyu.edu/~ritter/ritter2017machine.pdf) - With an appropriate choice of the reward function, reinforcement learning techniques can successfully handle the risk-averse case.
* [Ten Financial Applications of Machine Learning, 2018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197726) - Slides review few important financial ML applications.
* [FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, 2020](https://arxiv.org/abs/2011.09607) - Introduce a DRL library FinRL that facilitates beginners to expose themselves to quantitative finance and to develop their own stock trading strategies.
* [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) - Propose an ensemble strategy that employs deep reinforcement schemes to learn a stock trading strategy by maximizing investment return.

## Courses & Books & Blogs

* 🌟 [QuantResearch](https://github.com/letianzj/QuantResearch) ⭐ 2,807 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2023-08-26 - Quantitative analysis, strategies and backtests <https://letianzj.github.io/>
* [Train and Deploy a Serverless API to predict crypto prices](https://github.com/Paulescu/hands-on-train-and-deploy-ml) ⭐ 877 | 🐛 6 | 🌐 Python | 📅 2024-05-29 - In this tutorial you won't build an ML system that will make you rich. But you will master the MLOps frameworks and tools you need to build ML systems that, together with tons of experimentation, can take you there.
* [Advanced-Deep-Trading](https://github.com/Rachnog/Advanced-Deep-Trading) ⭐ 565 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2020-11-29 - Experiments based on "Advances in financial machine learning" book.
* [MLSys-NYU-2022](https://github.com/jacopotagliabue/MLSys-NYU-2022/tree/main) ⭐ 547 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2022-12-11 - Slides, scripts and materials for the Machine Learning in Finance course at NYU Tandon, 2022.
* [Mastering Python for Finance](https://github.com/jamesmawm/mastering-python-for-finance-second-edition) ⭐ 442 | 🐛 6 | 🌐 Jupyter Notebook | 📅 2026-02-03 - Sources codes for: Mastering Python for Finance, Second Edition.
* [NYU: Overview of Advanced Methods of Reinforcement Learning in Finance](https://www.coursera.org/learn/advanced-methods-reinforcement-learning-finance/home/welcome)
* [Udacity: Artificial Intelligence for Trading](https://www.udacity.com/course/ai-for-trading--nd880)
* [AI in Finance](https://cfte.education/) - Learn Fintech Online.
* [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos-ebook/dp/B079KLDW21/ref=sr_1_1?s=books\&ie=UTF8\&qid=1541717436\&sr=1-1) - Using advanced ML solutions to overcome real-world investment problems.
* [Build Financial Software with Generative AI](https://www.manning.com/books/build-financial-software-with-generative-ai?ar=false\&lpse=B&) - Book about how to build financial software hands-on using generative AI tools like ChatGPT and Copilot.
* [Financial AI in Practice](https://www.manning.com/books/financial-ai-in-practice) - A book about creating profitable, regulation-compliant financial applications.
* [Investing for Programmers](https://www.manning.com/books/investing-for-programmers) - A book about maximizing your portfolio, analyzing markets, and making data-driven investment decisions using Python and generative AI

## Strategies & Research

### Time Series Data

Price and Volume process with Technology Analysis Indices

* [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library) ⭐ 13,965 | 🐛 295 | 🌐 Jupyter Notebook | 📅 2026-01-30 - A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance.
* 🌟🌟 [stockpredictionai](https://github.com/borisbanushev/stockpredictionai) ⭐ 5,447 | 🐛 366 | 🌐 JavaScript | 📅 2025-08-19 - A complete process for predicting stock price movements.
* 🌟 [Ensemble-Strategy](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020) ⭐ 2,706 | 🐛 47 | 🌐 Python | 📅 2026-02-05 - Deep Reinforcement Learning for Automated Stock Trading.
* 🌟 [Personae](https://github.com/Ceruleanacg/Personae) ⭐ 1,397 | 🐛 9 | 🌐 Python | 📅 2018-11-29 - Implements and environment of Deep Reinforcement Learning & Supervised Learning for Quantitative Trading.
* [mlforecast](https://github.com/Nixtla/mlforecast) ⭐ 1,165 | 🐛 19 | 🌐 Python | 📅 2026-02-20 - Scalable machine learning based time series forecasting.
* [stock\_market\_reinforcement\_learning](https://github.com/kh-kim/stock_market_reinforcement_learning) ⭐ 799 | 🐛 18 | 🌐 Python | 📅 2016-12-23 - Stock market trading OpenAI Gym environment with Deep Reinforcement Learning using Keras.
* [Chaos Genius](https://github.com/chaos-genius/chaos_genius) ⚠️ Archived - ML powered analytics engine for outlier/anomaly detection and root cause analysis..
* [gym-trading](https://github.com/hackthemarket/gym-trading) ⭐ 713 | 🐛 6 | 🌐 Jupyter Notebook | 📅 2018-02-26 - Environment for reinforcement-learning algorithmic trading models.
* [deep\_rl\_trader](https://github.com/miroblog/deep_rl_trader) ⭐ 420 | 🐛 35 | 🌐 Python | 📅 2022-12-08 - Trading environment(OpenAI Gym) + DDQN (Keras-RL).
* [DeepLearningNotes](https://github.com/AlphaSmartDog/DeepLearningNotes) ⭐ 379 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2018-02-03 - Machine learning in quant analysis.
* [AutomatedStockTrading-DeepQ-Learning](https://github.com/sachink2010/AutomatedStockTrading-DeepQ-Learning) ⭐ 285 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2021-08-31 - Build a Deep Q-learning reinforcement agent model as automated trading robot.
* [tf\_deep\_rl\_trader](https://github.com/miroblog/tf_deep_rl_trader) ⭐ 254 | 🐛 36 | 🌐 Python | 📅 2022-12-08 - Trading environment(OpenAI Gym) + PPO(TensorForce).
* [trading-gym](https://github.com/6-Billionaires/trading-gym) ⭐ 233 | 🐛 28 | 🌐 Jupyter Notebook | 📅 2022-12-08 - Trading agent to train with episode of short term trading itself.
* [trading-rl](https://github.com/Kostis-S-Z/trading-rl) ⭐ 223 | 🐛 6 | 🌐 Python | 📅 2023-03-24 - Deep Reinforcement Learning for Financial Trading using Price Trailing.
* [zenbrain](https://github.com/carlos8f/zenbrain) ⭐ 51 | 🐛 2 | 🌐 CSS | 📅 2016-08-29 - A framework for machine-learning bots.
* [Quantitative-Trading](https://github.com/Ceruleanacg/Quantitative-Trading) ⭐ 37 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2018-05-09 - Papers and code implementing Quantitative-Trading.

### Portfolio Management

* [skfolio](https://github.com/skfolio/skfolio) ⭐ 1,874 | 🐛 24 | 🌐 Python | 📅 2026-02-10 - Python library for portfolio optimization built on top of scikit-learn.
* [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio) ⭐ 1,849 | 🐛 56 | 🌐 Python | 📅 2021-10-09 - A Deep Reinforcement Learning framework for the financial portfolio management problem.
* [DeepDow](https://github.com/jankrepl/deepdow) ⭐ 1,112 | 🐛 27 | 🌐 Python | 📅 2024-01-24 - Portfolio optimization with deep learning.
* [Deep-Reinforcement-Stock-Trading](https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading) ⭐ 688 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2024-11-06 - A light-weight deep reinforcement learning framework for portfolio management.
* [qtrader](https://github.com/filangel/qtrader) ⚠️ Archived - Reinforcement Learning for portfolio management.

### High Frequency Trading

* [High-Frequency-Trading-Model-with-IB](https://github.com/jamesmawm/High-Frequency-Trading-Model-with-IB) ⭐ 2,891 | 🐛 12 | 🌐 Python | 📅 2025-05-29 - A high-frequency trading model using Interactive Brokers API with pairs and mean-reversion.
* 🌟 [SGX-Full-OrderBook-Tick-Data-Trading-Strategy](https://github.com/rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy) ⭐ 2,218 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2022-08-27 - Solutions for high-frequency trading (HFT) strategies using data science approaches (Machine Learning) on Full Orderbook Tick Data.
* [HFT\_Bitcoin](https://github.com/ghgr/HFT_Bitcoin) ⭐ 168 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2017-08-21 - Analysis of High Frequency Trading on Bitcoin exchanges.

### Event Drive

* 🌟 [trump2cash](https://github.com/maxbbraun/trump2cash) ⚠️ Archived - A stock trading bot powered by Trump tweets.
* 🌟🌟 [stockpredictionai](https://github.com/borisbanushev/stockpredictionai) ⭐ 5,447 | 🐛 366 | 🌐 JavaScript | 📅 2025-08-19 - Complete process for predicting stock price movements.

### Crypto Currencies Strategies

* [tforce\_btc\_trader](https://github.com/lefnire/tforce_btc_trader) ⭐ 831 | 🐛 21 | 🌐 Jupyter Notebook | 📅 2019-02-13 - TensorForce Bitcoin trading bot.
* [LSTM-Crypto-Price-Prediction](https://github.com/SC4RECOIN/LSTM-Crypto-Price-Prediction) ⭐ 354 | 🐛 1 | 🌐 Python | 📅 2021-08-10 - Predicting price trends in crypto markets using an LSTM-RNN for trading.
* [gekkoga](https://github.com/gekkowarez/gekkoga) ⭐ 311 | 🐛 27 | 🌐 JavaScript | 📅 2019-02-02 - Genetic algorithm for solving optimization of trading strategies using Gekko.
* [bitcoin\_prediction](https://github.com/llSourcell/bitcoin_prediction) ⭐ 235 | 🐛 9 | 🌐 Jupyter Notebook | 📅 2018-02-01 - Code for "Bitcoin Prediction" by Siraj Raval on YouTube.
* [Tensorflow-NeuroEvolution-Trading-Bot](https://github.com/SC4RECOIN/Tensorflow-NeuroEvolution-Trading-Bot) ⭐ 163 | 🐛 4 | 🌐 Go | 📅 2021-03-09 - A population model that trade cyrpto and breed and mutate iteratively.
* [gekko-neuralnet](https://github.com/zschro/gekko-neuralnet) ⭐ 94 | 🐛 0 | 🌐 JavaScript | 📅 2020-07-16 - Neural network strategy for Gekko.
* [Gekko\_ANN\_Strategies](https://github.com/markchen8717/Gekko_ANN_Strategies) ⭐ 55 | 🐛 1 | 🌐 JavaScript | 📅 2023-08-25 - ANN trading strategies for the Gekko trading bot.

### Technical Analysis

* [quant-trading](https://github.com/je-suis-tm/quant-trading) ⭐ 9,191 | 🐛 3 | 🌐 Python | 📅 2024-04-14 - Python quantitative trading strategies.
* [crypto-signal](https://github.com/CryptoSignal/crypto-signal) ⭐ 5,458 | 🐛 57 | 🌐 Python | 📅 2024-07-07 - Automated crypto trading & technical analysis (TA) bot for Bittrex, Binance, GDAX, and more.
* [Gekko-Strategies](https://github.com/xFFFFF/Gekko-Strategies) ⭐ 1,413 | 🐛 17 | 🌐 JavaScript | 📅 2020-01-09 - Strategies to Gekko trading bot with backtests results and some useful tools.
* [Gekko-Bot-Resources](https://github.com/cloggy45/Gekko-Bot-Resources) ⚠️ Archived - Gekko bot resources.
* [forex.analytics](https://github.com/mkmarek/forex.analytics) ⚠️ Archived - Node.js native library performing technical analysis over an OHLC dataset with use of genetic algorithmv.
* [gekko\_tools](https://github.com/tommiehansen/gekko_tools) ⭐ 143 | 🐛 0 | 🌐 Shell | 📅 2020-02-04 - Gekko strategies, tools etc.
* [gekko\_trading\_stuff](https://github.com/thegamecat/gekko-trading-stuff) ⭐ 110 | 🐛 3 | 🌐 JavaScript | 📅 2018-04-02 - Awesome crypto currency trading platform.
* [gekko-gannswing](https://github.com/johndoe75/gekko-gannswing) ⭐ 71 | 🐛 2 | 🌐 JavaScript | 📅 2017-11-13 - Gann's Swing trade strategy for Gekko trade bot.
* [QTradeX](https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK) ⭐ 56 | 🐛 5 | 🌐 Python | 📅 2026-01-13 - A powerful and flexible Python framework for designing, backtesting, optimizing, and deploying algotrading bots
* [gekko HL](https://github.com/mounirlabaied/gekko-strat-hl) ⚠️ Archived - Calculate down peak and trade on.
* [Bitcoin\_MACD\_Strategy](https://github.com/VermeirJellen/Bitcoin_MACD_Strategy) ⭐ 10 | 🐛 0 | 🌐 R | 📅 2017-09-10 - Bitcoin MACD crossover trading strategy backtest.
* [gekko RSI\_WR](https://github.com/zzmike76/gekko) ⭐ 4 | 🐛 0 | 🌐 JavaScript | 📅 2018-03-03 - Gekko RSI\_WR strategies.
* [EthTradingAlgorithm](https://github.com/Philipid3s/EthTradingAlgorithm) ⭐ 4 | 🐛 0 | 🌐 Python | 📅 2018-06-23 - Ethereum trading algorithm using Python 3.5 and the library ZipLine.

### Lottery & Gamble

* [LotteryPredict](https://github.com/chengstone/LotteryPredict) ⭐ 412 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2019-06-10 - Use LSTM to predict lottery.

### Arbitrage

* [bitcoin-arbitrage](https://github.com/maxme/bitcoin-arbitrage) ⭐ 2,565 | 🐛 14 | 🌐 Python | 📅 2024-10-20 - Bitcoin arbitrage opportunity detector.
* [cryptocurrency-arbitrage](https://github.com/manu354/cryptocurrency-arbitrage) ⭐ 1,259 | 🐛 17 | 🌐 JavaScript | 📅 2022-05-15 - A crypto currency arbitrage opportunity calculator. Over 800 currencies and 50 markets.
* [r2](https://github.com/bitrinjani/r2) ⭐ 810 | 🐛 34 | 🌐 TypeScript | 📅 2023-04-19 - Automatic arbitrage trading system powered by Node.js + TypeScript.
* [ArbitrageBot](https://github.com/BatuhanUsluel/ArbitrageBot) ⭐ 176 | 🐛 0 | 🌐 Python | 📅 2017-09-10 - Arbitrage bot that currently works on bittrex & poloniex.
* [blackbird](https://github.com/butor/blackbird) - Long / short market-neutral strategy.

## Data Sources

#### Traditional Markets

* [Tushare](https://github.com/waditu/tushare) ⭐ 14,458 | 🐛 689 | 🌐 Python | 📅 2024-03-13 - Crawling historical data of Chinese stocks.
* [yahoo-finance](https://github.com/lukaszbanasiak/yahoo-finance) ⭐ 1,424 | 🐛 88 | 🌐 Python | 📅 2023-12-25 - Python module to get stock data from Yahoo! Finance.
* 🌟 [Quandl](https://www.quandl.com/tools/api) - Get millions of financial and economic dataset from hundreds of publishers via a single free API.
* [Financial Data](https://financialdata.net/) - Stock Market and Financial Data API.

#### Crypto Currencies

* [Gekko-Datasets](https://github.com/xFFFFF/Gekko-Datasets) ⭐ 174 | 🐛 11 | 🌐 Perl | 📅 2018-05-31 - Gekko trading bot dataset dumps. Download and use history files in SQLite format.
* [CryptoInscriber](https://github.com/Optixal/CryptoInscriber) ⭐ 52 | 🐛 0 | 🌐 Python | 📅 2018-03-17 - A live crypto currency historical trade data blotter. Download live historical trade data from any crypto exchange.
* [CoinPulse](https://github.com/soutone/coinpulse-python) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2026-01-09 - Python SDK for cryptocurrency portfolio tracking with real-time prices, P/L calculations, backtesting, and price alerts. Free tier: 25 req/hr.

#### News Data

* [WorldMonitor](https://github.com/koala73/worldmonitor) ⭐ 8,788 | 🐛 47 | 🌐 TypeScript | 📅 2026-02-21 - AI-powered news aggregation, geopolitical monitoring, and infrastructure tracking in a unified situational awareness interface.

#### Alternative Data

* [Pizzint](https://www.pizzint.watch/) - Pentagon Pizza Index (PizzINT) is a real-time Pentagon pizza tracker that visualizes unusual activity at Pentagon-area pizzerias. It highlights a signal that has historically aligned with late-night, high-tempo operations and breaking news.

## Research Tools

* [pyfolio](https://github.com/quantopian/pyfolio) ⭐ 6,228 | 🐛 166 | 🌐 Jupyter Notebook | 📅 2023-12-23 - Portfolio and risk analytics in Python.
* 🌟🌟 [TensorTrade](https://github.com/tensortrade-org/tensortrade) ⭐ 5,964 | 🐛 41 | 🌐 Python | 📅 2026-02-19 - Trade efficiently with reinforcement learning.
* [alphalens](https://github.com/quantopian/alphalens) ⭐ 4,143 | 🐛 49 | 🌐 Jupyter Notebook | 📅 2024-02-12 - Performance analysis of predictive (alpha) stock factors.
* [zvt](https://github.com/zvtvz/zvt) ⭐ 3,981 | 🐛 19 | 🌐 Python | 📅 2026-01-18 - Zero vector trader.
* [WFGY](https://github.com/onestardao/WFGY) ⭐ 1,489 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-02-21 – Open source framework for debugging and stress testing LLM agents and RAG pipelines. Includes a 16 mode failure map and long-horizon stress tests that are useful for financial research agents.
* [empyrical](https://github.com/quantopian/empyrical) ⭐ 1,457 | 🐛 37 | 🌐 Python | 📅 2024-07-26 - Common financial risk and performance metrics. Used by Zipline and pyfolio.
* [JAQS](https://github.com/quantOS-org/JAQS) ⭐ 633 | 🐛 45 | 🌐 Python | 📅 2019-04-25 - An open source quant strategies research platform.
* [Synthical](https://synthical.com) - AI-powered collaborative environment for Research.
* [ML-Quant](https://www.ml-quant.com/) - Quant resources from ArXiv (sanity), SSRN, RePec, Journals, Podcasts, Videos, and Blogs.

## Trading System

For Back Test & Live trading

### Traditional Market

**System**

* 🌟🌟🌟 [OpenBB](https://github.com/OpenBB-finance/OpenBB) ⭐ 60,546 | 🐛 60 | 🌐 Python | 📅 2026-02-21 - AI-powered opensource research and analytics workspace.
* [backtrader](https://github.com/backtrader/backtrader) ⭐ 20,472 | 🐛 56 | 🌐 Python | 📅 2024-08-19 - Python backtesting library for trading strategies.
* 🌟🌟 [zipline](https://github.com/quantopian/zipline) ⭐ 19,434 | 🐛 367 | 🌐 Python | 📅 2024-02-13 - A python algorithmic trading library.
* [lean](https://github.com/QuantConnect/Lean) ⭐ 16,637 | 🐛 242 | 🌐 C# | 📅 2026-02-20 - Algorithmic trading engine built for easy strategy research, backtesting and live trading.
* [rqalpha](https://github.com/ricequant/rqalpha) ⭐ 6,169 | 🐛 31 | 🌐 Python | 📅 2026-02-11 - A extendable, replaceable Python algorithmic backtest & trading framework.
* [kungfu](https://github.com/taurusai/kungfu) ⭐ 3,811 | 🐛 11 | 🌐 C++ | 📅 2024-05-02 - Kungfu Master trading system.
* 🌟 [TradingView](http://tradingview.com/) - Get real-time information and market insights.

**Combine & Rebuild**

* [pylivetrader](https://github.com/alpacahq/pylivetrader) ⭐ 681 | 🐛 20 | 🌐 Python | 📅 2022-10-04 - Python live trade execution library with zipline interface.
* [CoinMarketCapBacktesting](https://github.com/JimmyWuMadchester/CoinMarketCapBacktesting) ⚠️ Archived - As backtest frameworks for coin trading strategy.

### Crypto Currencies

* [abu](https://github.com/bbfamily/abu) ⭐ 16,227 | 🐛 3 | 🌐 Python | 📅 2026-01-24 - A quant trading system base on python.
* [zenbot](https://github.com/DeviaVir/zenbot) ⚠️ Archived - Command-line crypto currency trading bot using Node.js and MongoDB.
* [catalyst](https://github.com/enigmampc/catalyst) ⚠️ Archived - An algorithmic trading library for Crypto-Assets in python.
* [magic8bot](https://github.com/magic8bot/magic8bot) ⭐ 397 | 🐛 11 | 🌐 TypeScript | 📅 2023-03-04 - Crypto currency trading bot using Node.js and MongoDB.
* [bot18](https://github.com/carlos8f/bot18) ⭐ 201 | 🐛 12 | 🌐 HTML | 📅 2022-12-02 - High-frequency crypto currency trading bot developed by Zenbot.
* [QuantResearchDev](https://github.com/mounirlabaied/QuantResearchDev) ⚠️ Archived - Quant Research dev & Traders open source project.
* [MACD](https://github.com/sudoscripter/MACD) - Zenbot MACD Auto-Trader.

#### Plugins

* [Gekko-BacktestTool](https://github.com/xFFFFF/Gekko-BacktestTool) ⭐ 230 | 🐛 36 | 🌐 Perl | 📅 2020-03-14 - Batch backtest, import and strategy params optimalization for Gekko Trading Bot.
* [CoinMarketCapBacktesting](https://github.com/JimmyWuMadchester/CoinMarketCapBacktesting) ⚠️ Archived - Tests bt and Quantopian Zipline as backtesting frameworks for coin trading strategy.

## TA Lib

* [techan.js](https://github.com/andredumas/techan.js) ⭐ 2,438 | 🐛 103 | 🌐 JavaScript | 📅 2020-10-02 - A visual, technical analysis and charting (Candlestick, OHLC, indicators) library built on D3.
* [finta](https://github.com/peerchemist/finta) ⚠️ Archived - Common financial technical indicators implemented in Python-Pandas (70+ indicators).
* [pandas\_talib](https://github.com/femtotrader/pandas_talib) ⭐ 781 | 🐛 15 | 🌐 Python | 📅 2018-05-30 - A Python Pandas implementation of technical analysis indicators.
* [tulipnode](https://github.com/TulipCharts/tulipnode) ⭐ 513 | 🐛 18 | 🌐 JavaScript | 📅 2023-06-28 - Official Node.js wrapper for Tulip Indicators. Provides over 100 technical analysis overlay and indicator functions.

## Exchange API

Do it in real world!

* [IbPy](https://github.com/blampe/IbPy) ⚠️ Archived - Python API for the Interactive Brokers on-line trading system.
* [ctpwrapper](https://github.com/nooperpudd/ctpwrapper) ⭐ 561 | 🐛 19 | 🌐 Python | 📅 2025-05-03 - Shanghai future exchange CTP api.
* [PENDAX](https://github.com/CompendiumFi/PENDAX-SDK) ⭐ 48 | 🐛 1 | 📅 2024-05-09 - Javascript SDK for Trading/Data API and Websockets for cryptocurrency exchanges like FTX, FTXUS, OKX, Bybit, & More
* [HuobiFeeder](https://github.com/mmmaaaggg/HuobiFeeder) ⭐ 38 | 🐛 3 | 🌐 Python | 📅 2022-12-08 - Connect HUOBIPRO exchange, get market/historical data for ABAT trading platform backtest analysis and live trading.

### Framework

* [tf-quant-finance](https://github.com/google/tf-quant-finance) ⭐ 5,227 | 🐛 37 | 🌐 Python | 📅 2026-02-12 - High-performance TensorFlow library for quantitative finance.

### Visualizing

* [netron](https://github.com/lutzroeder/netron) ⭐ 32,415 | 🐛 19 | 🌐 JavaScript | 📅 2026-02-20 - Visualizer for deep learning and machine learning models.
* [playground](https://github.com/tensorflow/playground) ⭐ 12,781 | 🐛 139 | 🌐 TypeScript | 📅 2026-02-17 - Play with neural networks.
* [KLineChart](https://github.com/liihuu/KLineChart) ⭐ 3,574 | 🐛 52 | 🌐 TypeScript | 📅 2026-01-09 - Highly customizable professional lightweight financial charts

### GYM Environment

* 🌟 [TradingGym](https://github.com/Yvictor/TradingGym) ⭐ 1,840 | 🐛 11 | 🌐 Python | 📅 2024-02-11 - Trading and Backtesting environment for training reinforcement learning agent.
* [btgym](https://github.com/Kismuz/btgym) ⭐ 1,029 | 🐛 11 | 🌐 Python | 📅 2021-08-28 - Scalable, event-driven, deep-learning-friendly backtesting library.
* [TradzQAI](https://github.com/kkuette/TradzQAI) ⭐ 166 | 🐛 7 | 🌐 Python | 📅 2022-06-21 - Trading environment for RL agents, backtesting and training.

## Articles

* [The-Economist](https://github.com/nailperry-zd/The-Economist) ⭐ 3,786 | 🐛 20 | 📅 2023-06-23 - The Economist.
* [nyu-mlif-notes](https://github.com/wizardforcel/nyu-mlif-notes) ⭐ 104 | 🐛 0 | 📅 2018-10-24 - NYU machine learning in finance notes.
* [Using LSTMs to Turn Feelings Into Trades](https://www.quantopian.com/posts/watch-our-webinar-buying-happiness-using-lstms-to-turn-feelings-into-trades-now?utm_source=forum\&utm_medium=twitter\&utm_campaign=sentiment-analysis)

## Others

* [gekko-quasar-ui](https://github.com/H256/gekko-quasar-ui) ⚠️ Archived - An UI port for gekko trading bot using Quasar framework.
* [zipline-tensorboard](https://github.com/jimgoo/zipline-tensorboard) ⭐ 107 | 🐛 1 | 🌐 Python | 📅 2022-10-26 - TensorBoard as a Zipline dashboard.
* [Floom](https://github.com/FloomAI/Floom) ⭐ 46 | 🐛 0 | 🌐 C# | 📅 2024-11-17 AI gateway and marketplace for developers, enables streamlined integration and least volatile approach of AI features into products

#### Other Resource

* [awesome-quant](https://github.com/wilsonfreitas/awesome-quant) ⭐ 24,307 | 🐛 37 | 🌐 Jupyter Notebook | 📅 2026-02-21 - A curated list of insanely awesome libraries, packages and resources for Quants (Quantitative Finance).
* 🌟🌟🌟 [Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) ⚠️ Archived - Stock-Prediction-Models, Gathers machine learning and deep learning models for Stock forecasting, included trading bots and simulations.
* 🌟🌟 [Financial Machine Learning](https://github.com/firmai/financial-machine-learning) ⭐ 8,401 | 🐛 9 | 🌐 Python | 📅 2025-01-03 - A curated list of practical financial machine learning (FinML) tools and applications. This collection is primarily in Python.
* 🌟 [Awesome-Quant-Machine-Learning-Trading](https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading) ⭐ 3,469 | 🐛 1 | 📅 2025-05-21 - Quant / Algorithm trading resources with an emphasis on Machine Learning.
* [FinancePy](https://github.com/domokane/FinancePy) ⭐ 2,778 | 🐛 50 | 🌐 Jupyter Notebook | 📅 2026-01-27 - A Python Finance Library that focuses on the pricing and risk-management of Financial Derivatives, including fixed-income, equity, FX and credit derivatives.
* [Explore Finance Service Libraries & Projects](https://kandi.openweaver.com/explore/financial-services#Top-Authors) - Explore a curated list of Fintech popular & new libraries, top authors, trending project kits, discussions, tutorials & learning resources on kandi.
* [AgentMarket](https://agentmarket.cloud) - B2A marketplace for AI agents. 189 listings, 28M+ real energy data records, LangChain/MCP integration.
