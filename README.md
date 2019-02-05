SNMoE_Matlab


A Matlab/Octave toolbox for modeling, sampling, inference, and clustering heteregenous data with the Skew-Normal Mixture-of-Experts (SNMoE) model.

The code can be run on simulated data and some provided benchmarks (please look to the functions main... for more details):

run 'main_univ_NMoE_SNMoE_TMoE_STMoE.m' for some simulated data
run 'main_univ_NMoE_TMoE_STMoE_RealData.m' for some benchmarks

Please cite the following papers for this code:

``` 
@InProceedings{Chamroukhi-SNMoE-IJCNN-2016,
    Author         = {F. Chamroukhi},
    booktitle  = {The International Joint Conference on Neural Networks (IJCNN)},
    Address = {Vancouver, Canada},
    Title          = {Skew-Normal Mixture of Experts},
    Year           = {2016},
	Month = {July},
	url = {https://chamroukhi.com/papers/Chamroukhi-SNMoE-IJCNN2016.pdf},
	slides = {./conf-presentations/FChamroukhi-IJCNN-2016-Talk.pdf},
	software =  {https://github.com/fchamroukhi/SNMoE_Matlab}
	}
  
@article{Chamroukhi-NNMoE-2015,
	Author = {F. Chamroukhi},
	eprint = {arXiv:1506.06707},
	Title = {Non-Normal Mixtures of Experts},
	Volume = {},
	url= {http://arxiv.org/pdf/1506.06707.pdf},
	month = {July},
	Year = {2015},
	note = {Report (61 pages)}
}

@article{NguyenChamroukhi-MoE,
	Author = {Hien D. Nguyen and Faicel Chamroukhi},
	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
	Title = {Practical and theoretical aspects of mixture-of-experts modeling: An overview},
publisher = {Wiley Periodicals, Inc},
issn = {1942-4795},
doi = {10.1002/widm.1246},
pages = {e1246--n/a},
keywords = {classification, clustering, mixture models, mixture of experts, neural networks},
	Month = {Feb},
Year = {2018},
url = {https://chamroukhi.com/papers/Nguyen-Chamroukhi-MoE-DMKD-2018}
}
```
Developed and written by Faicel Chamroukhi
(c) F. Chamroukhi (2015)
