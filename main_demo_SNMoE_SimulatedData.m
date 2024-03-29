%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% >>>> skew-normal mixture-of-experts (SNMoE) <<<<<
% 
% SNMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
% heterogeneous data with the Skew-Normal Mixture-of-Experts (SNMoE) model.
% 
% SNMoE provides a flexible modeling framework for heterogenous data with possibly skewed
% distributions to generalize the standard Normal mixture of expert model. SNMoE consists of a
% mixture of K skew-Normal expert regressors network (of degree p) gated by a softmax gating network
% (with regression degree q) and is represented by 
% - The gating net. parameters $\alpha$'s of the softmax net.
% - The experts network parameters: The location parameters (regression coefficients) $\beta$'s,
% scale parameters $\sigma$'s, and the skewness parameters $\lambda$'s. 
% SNMoE thus generalises  mixtures of (normal, skew-normal) distributions and mixtures of
% regressions with these distributions. For example, when $q=0$, we retrieve mixtures of
% (skew-normal, or normal) regressions, and when both $p=0$ and $q=0$, it is a mixture of
% (skew-normal, or normal) distributions. It also reduces to the standard (normal, skew-normal)
% distribution when we only use a single expert (K=1).
% 
% Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
% algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
% illustrate the use of the model in model-based clustering of heteregenous regression data and in
% fitting non-linear regression functions. Real-world data examples of tone perception for musical
% data analysis, and the one of temperature anomalies for the analysis of climate change data, are
% also provided as application of the model.
% 
% To run it on the provided examples, please run "main_demo_SNMoE_SimulatedData.m" or
% "main_demo_SNMoE_RealData.m"
% 
%% Please cite the code and the following papers when using this code: 
% - F. Chamroukhi. Skew-Normal Mixture of Experts., July, 2016, The International Joint Conference on Neural Networks. 
% - F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015 
%
% (c)Introduced and written by Faicel Chamroukhi (may 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc; 
%%  sample drawn from the model:
n = 500;
% some specified parameters for the model
Alphak = [0, 8]';
Betak = [0 0;
    -2.5 2.5];
Sigmak = [1, 1];%the standard deviations
Lambdak = [3, 5];
Nuk = [5, 7];

% sample the data
x = linspace(-1, 1, n);
[y, klas, stats, Z] = sample_univ_SNMoE(Alphak, Betak, Sigmak, Lambdak, x);

%% add (or no) outliers (to show that the model is not robust to outliers, unlike TMoE and STMoE
WithOutliers = 0; % to generate a sample with outliers
% if outliers
if WithOutliers
    rate = 0.05;%amount of outliers in the data
    No = round(length(y)*rate);
    outilers = -1.5 + 2*rand(No,1);
    tmp = randperm(length(y));
    Indout = tmp(1:No);
    y(Indout) = -5; %outilers;
end

%% model learning
% model structure setting
K = 2; % number of experts
p = 1; % degree the polynomial regressors (Experts Net)
q = 1; % degree of the logstic regression (gating Net)

% EM options setting
nb_EM_runs = 3;
max_iter_EM = 1500;
threshold = 1e-6;
verbose_EM = 1;
verbose_NR = 0;

%% learn the model from the sampled data
SNMoE =  learn_SNMoE_EM(y, x, K, p, q, nb_EM_runs, max_iter_EM, threshold, verbose_EM, verbose_NR);

disp('- fit completed --')
%% plot of the results
show_SNMoE_results(x, y, SNMoE, klas, stats)
