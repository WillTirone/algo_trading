# algo_trading
Repository for our final project in MA 585 (Algorithmic Trading) taken at Duke University in Spring 2023.

# Abstract
This paper presents a novel approach to algorithmic trading that combines fundamental universe selection,
regression using Gradient Boosting, linear programming to choose portfolio weights, and rigorous risk management
controls to optimize portfolio performance. Our methodology involves selecting a subset of stocks
from a large universe of U.S. equities based on fundamental criteria such as earnings, debt, and valuation
metrics. We then use Gradient Boosting to predct returns the selected stocks as either buys or sells based
on their expected performance. To construct an optimal portfolio, we use a linear programming model
that maximizes the Sharpe Ratio while imposing constraints on risk and exposure. Our results show that
the proposed approach outperforms the market during periods of economic crisis but underperforms during
bull markets. Furthermore, we demonstrate the effectiveness of our risk controls in mitigating tail risk and
preserving capital during market downturns. Overall, our findings suggest that incorporating fundamental
universe selection, Gradient Boosting, linear programming, and risk controls can lead to superior investment
outcomes with algorithmic trading during bear markets.
