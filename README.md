>>>> skew-normal mixture-of-experts (SNMoE) <<<<<

SNMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
heterogeneous data with the Skew-Normal Mixture-of-Experts (SNMoE) model.

SNMoE provides a flexible modeling framework for heterogenous data with possibly
skewed distributions to generalize the standard Normal mixture of expert model. SNMoE consists of a
mixture of K skew-Normal expert regressors network (of degree p) gated by a softmax gating network
(with regression degree q) and is represented by 
- The gating net. parameters $\alpha$'s of the softmax net. 
- The experts network parameters: The location parameters (regression coefficients)
$\beta$'s, scale parameters $\sigma$'s, and the skewness parameters $\lambda$'s. SNMoE thus generalises  mixtures of (normal, skew-normal) distributions and mixtures of regressions with these distributions. For example, when
$q=0$, we retrieve mixtures of (skew-normal, or normal) regressions, and when both
$p=0$ and $q=0$, it is a mixture of (skew-normal, or normal) distributions. It also
reduces to the standard (normal, skew-normal) distribution when we only use a
single expert (K=1).

Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
illustrate the use of the model in model-based clustering of heteregenous regression data and in
fitting non-linear regression functions. Real-world data examples of tone perception for musical
data analysis, and the one of temperature anomalies for the analysis of climate change data, are
also provided as application of the model.

To run it on the provided examples, please run "main_demo_SNMoE_SimulatedData.m" or "main_demo_SNMoE_RealData.m"

``` 
Please cite the code and the following papers when using this code: 
- F. Chamroukhi. Skew-Normal Mixture of Experts., July, 2016, The International Joint Conference on Neural Networks.
- F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015
``` 
(c) Introduced and written by Faicel Chamroukhi (may 2015)