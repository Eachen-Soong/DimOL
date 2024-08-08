# DimOL Reburral (NIPS 24)



<!-- 要做的实验：
1. 关于不同的p的ablation exp
2. 增加4.7的内容
3. 针对最后一个reviewer，看看能不能在小规模模型上使用SimAug dataset 说明dimensional awareness体现在何处
4. 不同的seed

要修改的文章内容：
1. 4.5: clarification on what MSE-OOD and MSE-Few settings are -->

## Author Rebuttal

We thank all reviewers for the constructive comments. In response, we have extended our experiments and present new results in this rebuttal, specifically including:
1. Running the training process with multiple random seeds and reporting the results along with standard deviations.
2. Evaluating the number of extra MLP parameters that should be added to the baseline model to achieve comparable performance to DimOL with a single ProdLayer.
3. Conducting ablation studies on the effects of different $p$ values, i.e., the number of product terms per ProdLayer.

Notably, we make two corrections regarding the experimental setup in the paper:
- **Correction 1:** After careful examination, we found that the baselines we compared may not be adequate. In experiments conducted on the Burgers and TorusVisForce datasets, we replaced the original channel-mixing MLP with the ProdLayer; whereas our baseline model did not incorporate channel mixing. To ensure a fair comparison, in this rebuttal, we further evaluate our approach against FNO-based models that incorporate the channel-mixing MLP. 
    - We present the updated results below. On the TorusVisForce dataset, the performance gain of DimOL is about **5.5% compared to T-FNO (w/ channel mixing)**, suggesting that the initial 74.3% improvement in Table 3 is primarily due to the channel-mixing trick. 
    - Using channel-mixing trick doesn't necessarily reduce the error in our experiment, so some promotion ratio still holds.
    - For LSM and GNOT, the original conclusion remains valid, as the baseline results presented in the manuscript were obtained using models that do incorporate channel mixing.
    - For all DimOL models without marking the hypermater p in paper, we adopt the default value p=2 (which may not be optimal, but can always gain improvement on loss).

(1) Promotion against FNO (w/ chan. mix) on Burgers: **48.4%**.
| Burgers | FNO (w/o chan. mix) | FNO (w/ chan. mix) | FNO + ProdLayer (p=2) |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE (1e-3) |       2.225             |       2.342             |            1.147     |

<!-- (2) Promotion against T-FNO (w/ chan. mix) on TorusLi: 1.5%.

| TorusVisLi (T=10) | T-FNO (w/o chan. mix) | T-FNO (w/ chan. mix) | T-FNO + ProdLayer |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE |          0.1448            |            0.1425            |          0.1404       | -->

(2) Promotion against T-FNO (w/ chan. mix) on TorusVisForce: **11.5%** (T=4); **5.5%** (T=10).

| TorusVisForce (T=4) | T-FNO (w/o chan. mix) | T-FNO (w/ chan. mix) | T-FNO + ProdLayer |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE (1e-2) |            1.193          |          1.090           |        0.966      |

| TorusVisForce (T=10) | T-FNO (w/o chan. mix) | T-FNO (w/ chan. mix) | T-FNO + ProdLayer |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE (1e-2) |       4.649             |         1.717            |          1.622       |

(3) Promotion against T-FNO (w/ chan. mix) on Few-shot TorusVisForce: **16.5%**.

| Few-shot TorusVisForce (T=4) | T-FNO (w/o chan. mix) | T-FNO (w/ chan. mix) | T-FNO + ProdLayer |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE (1e-2) |         2.037              |         2.104              |        1.757       |



- **Correction 2:** Another issue is about the Burgers experiment, where the test set should not be sub-sampled. We have corrected this problem and present the new results above. As shown, the performance gain is **48.4%**, which is significantly greater than the original 6.7% result shown in Table 1.


**Ablation study on ProdLayer dimensions:** We here provide the ablation study results on different values of $p$ in the ProdLayer, where $p=0$ is equivalent to using a 2-layer MLP as the channel-mixing layers. We can see that Using an arbitary value $p=1,2$ consistently improves the performance of base models that do not incorporate the ProdLayer ($p=0$).
- Dataset: Burgers. Metric: MSE (1e-3). Please note that the results differ from those originally presented in the paper due to Correction 2 mentioned above.

| Base model | w/o chan. mix | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| FNO | 2.225 | 2.342 | 1.130 | 1.147 | 1.163 | 1.059 | 1.071 |

- Dataset: DarcyFlow. Metric: MSE

| Base model | w/o chan. mix | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| T-FNO       | 10.83 | 7.540 | 7.192 | 6.919 | 7.367 | 6.613 | 6.807 |
| LSM         | / | 2.892 | 2.823 | 2.713 | 2.632 | 2.716 | 2.720 |

<!-- | FNO         | 5.073 | 5.889 | 5.689 | 5.726 | 6.024 | 5.641 | 5.487 | -->

-  Dataset: TorusLi (T=10). Metric: MSE

| Base model | w/o chan. mix | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| T-FNO       | 0.1448 | 0.1425 | 0.1396 | 0.1404 | 0.1384 | 0.1398 | 0.1398 |
| LSM         | / | 0.1824 | 0.1810 | 0.1806 | 0.1857 | 0.1828 | 0.1859 |

<!-- | FNO         | 0.1496 | 0.1404 | 0.1399 | 0.1376 | 0.1393 | 0.1383 | 0.1397 | -->

<!-- - Dataset: Few-shot TorusVisForce (T=4). Metric: MSE (1e-2)

| Base model | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| T-FNO       | 2.037 | 2.104 | 1.759 | 1.757 | 1.733 | 1.752 | 1.690 |
| LSM         | / | 8.925 | 8.553 | 8.605 | 8.675 | 8.557 | 8.692 | -->

<!-- Please check out the uploaded PDF file for the visualized results!  -->


## Responses to Reviewer Vt4z

### Rebuttal towards Weaknesses:
> 1. Lacking analysis on equation-constrained modelling or physics-informed learning techniques (slightly beyond the scope of the work but would be greatly beneficial if could be addressed)

In this work, we argue that equation-constrained modelling should not be considered as Operator Learning after all. Traditional Physics-Informed Neural Networks (PINNs) can only approximate functions rather than operators.

However, it is indeed beneficial to analyze the Physics Informed Neural Operators (PINO) [1], which can be understood as a combination of neural operators and equation-constrained methods. We will add these analyses in our follow-up study. Thank you for your valuable advice!

[1] Li et al. "Physics-informed neural operator for learning partial differential equations." ACM/JMS Journal of Data Science 1.3 (2024): 1-27.

> 2. Lacking discussion of potential futures about DimOL other than ProdLayer

In Appendix A in the original manuscript, we tested a quadratic layer (QuadLayer) against ProdLayer, and found that ProdLayer consistently performs better. 

Are there other options for achieving DimOL? The most recently released Kolmogorov-Arnold Networks (KAN)[2] can be one possibility. The latest version of KAN actually employs product encoding similar to ProdLayer, making it a potentially suitable choice. However, we tested KAN on some PDE toy datasets, and it doesn't seem to perform well with Neural Operators. 

Nonetheless, it is still possible that some other networks might be capable of achieving dimensional alignment for variables and could potentially outperform baseline models adapted with ProdLayers.

[2] Liu et al. "KAN: Kolmogorov-Arnold networks." arXiv preprint arXiv:2404.19756 (2024).

> 3. On how could DimOL achieve scientific discovery

In Section 4.7, we meant that if we can have a high-speed camera to capture the waveform of the Burgers System, then FNO+DimOL could effectively regress the evolutional equation of the system. 

However, this approach has not yet been demonstrated to be suitable for datasets with lower time resolution. The system's evolution may progress too far within the interval between frames, as seen in the Burgers dataset used in Section 4.1. In our follow-up studies, we plan to include analyses of other dynamic systems. 

> 4. Presentations/Grammar/Typos Errors

Thank you for reminding us! We will make the corrections in the revised paper.

### Q&A
> 1. (i) Could you provide additional information on how to choose a suitable $p$ on a suite of scenarios? (ii) What are the failure modes of $p$?

(i) How to choose a suitable $p$?

We conduct further ablation studies on different values of $p$ from [0, 1, 2, 4, 8, 16], where $p=0$ is equivalent to using a 2-layer MLP as the channel-mixing layers. Please refer to our General Response for the experimental results. From the results across various datasets, we can observe that:
- Using an arbitrary value $p$ greater than 0 consistently improves the performance of base models that do not incorporate the ProdLayer ($p=0$). 
- In most cases, $p$ can be simply set to 2. An increased number of $p$ does not typically yield further improvements, demonstrating the efficiency of our approach.

(ii) What are the failure modes of $p$?

It is noteworthy that the dimensional awareness technique is only beneficial when working with datasets where physical quantities are part of the input. If the dataset contains only geometry without any associated physical quantities, the dimensional awareness doesn't make sense. Consequently, the model is unlikely to benefit from the ProdLayer in such cases. This is actually discussed in Section 4.4 of the paper.

> 2. Were Neural Operator models on general or irregular geometries considered in the evaluation of the model? 
<!-- > (I am pleased to see the results on GNOT (a Transformer based model) with irregular meshes) -->

Existing datasets with irregular geometries used in previous operator learning literature are primarily mesh-only datasets that lack physical quantity inputs except for the Inductor2D dataset used in the work of GNOT. In Table 5 of Section 4.4, we have evaluated GNOT+ProdLayer on this dataset and observed a 4.8% performance gain compared with the results of GNOT.

NOTE: The title of Table 4 should be "Model performance on the Inductor2D dataset." rather than "... on the TorisLi dataset".

## Responses to Reviewer s4Eo

### Rebuttal towards Weaknesses:

The main concern is about whether the model can also work in datasets with more kinds of physics quantities.

Indeed, currently, the most complex dataset we have tested is Inductor2D, which involves 11 global scalar quantities. In contrast, the TorusVisForce dataset consists of 2 field variables and 1 scalar variable as input.

If there are other suggested datasets, we would also love to test DimOL on them. In our concept, DimOL is specifically aimed at handling multiple physical quantities.

### Q&A
> 1. How does DimOL handle PDEs where the dimensional relationships are not clearly defined or are subject to debate in the scientific community? Can it adapt to different formulations of the same physical problem?

In fact, in physics, most datasets would have clear dimensional relationships, since the data are collected by sensors and have measuring units. However, different formulations can be subject to debate in the scientific community. For example, in the Navier-Stokes 2D problem, one formulation is to directly use the velocity field, while another formulation is to use the vorticity field. Basically, the velocity field can be transformed into a vorticity field through a curl operator, while the vorticity field can also be derived from the velocity field given boundary values, both of which are representable by a single layer of FNO block. So if the different formulations can be converted to each other with some operators, then DimOL can possibly handle them.

> 2. The paper shows impressive few-shot learning results. Can you provide more insight into how DimOL achieves this, and are there theoretical guarantees on the sample efficiency?

As detailed in our **General Response**, we have updated the T-FNO baseline results under the few-shot learning setup on TorusVisForce. The results are shown below, where DimOL achieves an approximately **16.5%** performance gain compared to T-FNO (w/ channel mixing).

| Few-shot TorusVisForce (T=4) | T-FNO (w/o chan. mix) | T-FNO (w/ chan. mix) | T-FNO + ProdLayer |
| ---------- | ------------------------ | ----------------------- | --------------- |
|  MSE (1e-2) |         2.037              |         2.104              |        1.757       |

The performance gain of DimOL is likely attributed to its ability to handle global variables and different input variables. More concretely, a diffusion term can be represented as a latent variable multiplied by viscosity, allowing DimOL models to express this relationship directly, in contrast to traditional MLP channel mixing. However, when the input consists of only one channel, such as in TorusLi, the performance gain of DimOL on few-shot datasets would be significantly lower.

Regarding theoretical guarantees on sample efficiency, we unfortunately have not yet investigated how to prove this mathematically.

> 3. How does the ProdLayer interact with the spectral properties of FNOs? Does it affect the model's ability to represent different frequency components of the solution?

Actually, in our implementation, we did not modify the Spectral Convloution Layers of FNO, instead, we used the ProdLayer in-between Spectral Convloution Layers as channel mixing. This makes the channel-mixing MLP more physics-aware, as analyzed in Section 4.7, the spectral layer can symbolically represent the 1st and 2nd orders of derivatives, which can be deemed as a major property of the interaction between ProdLayer and Spectral Convolutions.

In terms of the ability to represent different frequency components, the DimOL models can be viewed as an enhanced version of their base model, but with stronger representabilities. Specifically, when the slices of parameters corresponding to the input dimension of the products are fixed to 0, the DimOL models would reduce to the base models.

> 4. Can DimOL be used for multi-physics problems where different subsystems might have distinct dimensional structures? How would you handle the interfaces between these subsystems?

Absolutely! The more physical quantities involved, the more beneficial the DimOL techniques become, as DimOL is fundamentally designed to deal with distinct dimensional structures. We would be happy to provide additional experimental results on any suggested multi-physics datasets if needed.

> 5. Does DimOL offer any advantages in working with dimensionless quantities?

Yes. Even when the quantities are dimensionless, the intrinsic ability to represent sum-of-product terms is crucial, as it facilitates the decoupling of relationships between variables.

## Responses to Reviewer qeUh

<!-- ### Q&A -->
> 1. Section 4.4 discusses the applicability of ProdLayer for irregular meshes and the Inductor2D dataset. However, the results for this dataset are never presented in the paper. 

Yes, Table 4 is indeed for Inductor2D. We are truly sorry for the mistake and will correct it in the revised paper. Thank you for your understanding!

> 2. In Figure 1-b, I believe the skip connection on the left should be connected to the addition block.

Yes, we will fix it in the revision. Thanks again!

> 3. In section 4.5 and Table 5, I assume MSE-OOD is the zero-shot performance. I would appreciate a clarification on what MSE-OOD and MSE-Few settings are (i.e. number of few-shot samples, etc.). 
<!-- > Authors may also consider other Lie Symmetries studied in [1] for further experiments on OOD samples. -->
The MSE-Few in the paper is the loss of a subset of TorusVisForce dataset, containing 200 pieces of data with the highest viscosity values (shuffled). The viscosity range for this subset is [8.2e-05, 1e-4]. We used 100 pieces of that for trainind and other 100 pieces for testing.

MSE-OOD, on the other hand, is the loss of the test set containing 100 pieces of data with the lowest viscosity values, whose range is [8.2e-05, 1e-4].

> 4. Some aspects of the proposed block need clarification. I assume is the number of channels. If that is the case, It's not clear how and are divided and with what dimensions. In the experiments, it seems like is always except for TorisVisForce LSM (Table 5). Also, does have the same dimension as ?

Clarification: all other are with p=2 by default.

## Responses to Reviewer hKDb

<!-- The ProdLayer parameterization seems promising, but in general the authors seem to overclaim their contributions.

The authors propose "dimension-aware operator learning" and claim that their ProdLayer parameterization is motivated from dimensional analysis. However, it seems like the extent of the connection is the justification of the sum-of-products parameterization. Beyond this, there is no evidence that the ProdLayer is doing anything like dimensional analysis, so it seems to be overclaiming to describe a model with ProdLayer as "dimension-aware".

The main connection to dimensional awareness with their proposed method seems to be Section 4.7, but the results are unconvincing. More details about this set of experiments could help explain the connections the authors suggest. -->

### Rebuttal towards Weaknesses:

> On the claim of 'dimension-aware'.

We have now added new experiments on the TorusVisForce dataset to showcase why the model is dimension-aware:

In Section 4.5, we introduced a method to generate 'similar' datasets from training data through dimensional analysis. We will refer to this generating process as a 'similar transformation'.

If we consider the ground-truth operator that the Neural Operator method needs to approximate, the invariance of similar transformation is a crucial property according to dimensional analysis. However, if we view this dataset from a traditional machine learning perspective (without accounting for operator learning), the generated data can be regarded as out-of-distribution. 

We present the specific experimental configurations and corresponding results below. As shown, the DimOL models consistently achieve **over 11% improvements** compared to LSM on OOD data generated through 'similar transformations', which exceeds the performance gain observed on the in-distribution test set (3.3%).

<!-- Our experiments showed that on completely OOD datasets for comparison, the boost would be only around 3% from baseline;on the other hand, on similar datasets, DimOL models can still remain the boost of MSE error (T-FNO+DimOL: around 10%) on similar datasets as the validation dataset within distribution compared to baselines. -->

This property is particularly beneficial when combining machine learning with traditional dimensional analysis. Specifically, it allows us to collect data on smaller-scale simulations and then transfer the model onto real-world normal-scale data, which is more challenging to acquire than small-scale simulations.

This will be introduced as a new section of the paper. Again, thanks for pointing out the weakness of our work!

#### Experimental Settings:

TorusVisForceTopMu: This is a subset of the TorusVisForce dataset, containing 200 pieces of data with the highest viscosity values (shuffled). The viscosity range for this subset is [8.2e-05, 1e-4].

<!-- TorusVisForceBottomMu: The subset of TorusVisForce, with 100 pieces of data with bottom viscosity (shuffled), mu range: [1e-05, 1.8e-5] -->

<!-- TrainSet: the first 100 samples of TorusVisForceTopMu

ID: In-Distribution Dataset, the last 100 samples of TorusVisForceTopMu

OOD: In-Distribution Dataset, TorusVisForceBottomMu

(k=n) Dataset: Similar Dataset generated by train dataset with scaling coefficient k=n

All datasets are tested zero-shot. -->

Train/Test Splits:
- Training: The first 100 samples of TorusVisForceTopMu.
- In-set testing: The in-distribution test set, the last 100 samples of TorusVisForceTopMu.
- OOD testing: Out-of-distribution test set generated by transforming the training data using a scaling coefficient $k$.

|  TorusVisForceTopMu (T=10) | MSE (In-set) | MSE (OOD, k=4) | MSE (OOD, k=16) | 
| ------------------ | -------- | --------- | ---------- | 
| LSM       | 0.08304  | 0.01427   | 0.01500    | 
| LSM + ProdLayer (p=2)        | 0.08031  | 0.01262   | 0.01334    | 
| Promotion | 3.3%     | 11.6%     | 11.1%      | 


<!-- |                    | MSE (ID) | MSE (k=4) | MSE (k=16) | MSE (OOD) |
| ------------------ | -------- | --------- | ---------- | --------- |
| T-FNO (no mixing)  | 0.02354  | 0.01158   | 0.01246    | 0.08975   |
| T-FNO (p=0)        | 0.01625  | 0.01005   | 0.01062    | 0.08000   |
| T-FNO (p=2)        | 0.01456  | 0.00911   | 0.00984    | 0.07828   |
| T-FNO (p=2)'s gain | 10.4%    | 9.4%      | 7.3%       | 2.2%      |
| LSM   (p=0)        | 0.08304  | 0.01427   | 0.01500    | 0.1639    |
| LSM   (p=2)        | 0.08031  | 0.01262   | 0.01334    | 0.1604    |
| LSM   (p=2)'s gain | 3.3%     | 11.6%     | 11.1%      | 2.1%      | -->

<!-- Note: for k=4 and k=16 test-sets, we use the model where MSE (ID) is lowest.  -->

### Q&A
> 1. The biggest gains from adding the ProdLayer seems to be on the TorisVisForce dataset with T-FNO. Is there a reason why we might expect this model and dataset seems to benefit so much? 

After careful examination, we found that the 74.3% performance gain is primarily attributable to the use of a baseline T-FNO model without the channel-mixing MLP (which is an optional component in the official T-FNO codebase). In this rebuttal, we conduct further experiments by incorporating the channel-mixing MLP (which has more parameters than ProdLayer) into the baseline model. Please refer to our **General Response (Correction 1)** for the updated results.

> 2. The main question I have is about the consistency/significance of the performance boosts with the ProdLayer. How do the performance gains compare to e.g. the variance in performance from different initialization via different random seeds?

First, we would like to summarize the performance gain achieved by using ProdLayer in comparison to FNO-based baseline models that incorporate channel-mixing MLPs. The average improvement across the 6 benchmarks is approximately **15%**. 

It is noteworthy that, since we applied the MLP (which has more parameters than ProdLayer) in the baseline model, having a 15% performance gain without increasing the number of parameters is a remarkable achievement.

| Burgers | DarcyFlow | TorusLi | TorusVisForce (T=4) | TorusVisForce (T=10) | Few-shot TorusVisForce | Average |
| ------- | --------- | ------- | ------------------- | -------------------- | ---------------------- | ------- |
| 48.4%   | 6.9%      | 1.5%    | 11.5%               | 5.5%                 | 16.5%                  | 15.1%   |

Note: The TorusLi data has only one input channel, while the performance gain of DimOL is largely attributed to its ability to handle relationships between different input variables. This explains why DimOL achieves only a 1.5% improvement on this dataset.

Further, as suggested by the reviewer, we train the models 4 times using random initialization seeds, and observe a standard deviation much smaller than the improvement. Below we would show the experiment detail of Burgers and TorusVisForce, whose improvement are respectively 43.3% and 5.5%. The performance gain of DimOL against different compared models remains consistent. According to the experiments below, the DimOL models are actually more stable than baselines.


| Burgers MSE(1e-3)       | 1     | 2     | 3     | 4     | mean  | std/mean |
| ---------------------------- | ----- | ----- | ----- | ----- | ----- | -------- |
| FNO (p=0)                    | 2.264 | 2.346 | 2.476 | 2.280 | 2.342 | 3.6%     |
| FNO (p=2)                    | 1.126 | 1.184 | 1.141 | 1.139 | 1.147 | 1.9%     |


| TorusVisForce MSE(1e-2) | 1     | 2     | 3     | 4     | mean  | std/mean |
| -                            | -     | -     | -     | -     | -     | -        |
| T-FNO (p=0)                  | 1.744 | 1.737 | 1.673 | 1.715 | 1.717 | 1.6%     |
| T-FNO (p=2)                  | 1.634 | 1.638 | 1.621 | 1.627 | 1.630 | 0.4%     |

Thanks for your advice to improve the experiment! 



> 3. Models with ProdLayer have slightly more parameters than the ones without (Tables 1, 2). Is it possible that the slight increase in size explains the boost in performance?

First, as mentioned earlier, the ProdLayer introduces an average performance gain of 15% compared to existing FNO-based models that incorporate channel-mixing MLPs with a similar number of parameters.

Additionally, per the reviewer's request, we have tested the performance of FNO and T-FNO baseline models with different numbers of MLP layers, within the range $L_{MLP} \in$ \{2, 3, 4, 5, 6\}. The results indicate that a 2 or 3 layer MLP would the best choice for the baseline. Baselines with 4 or more layers have a larger number of parameters than those with 2 layers, while simply increasing the parameter count in the MLP of FNO-based baselines does not necessarily lead to improved performance. In TorusVisForce case, though baseline with $L_{MLP}=3$ outperforms the one with $L_{MLP}=2$, the parameter amount is slightly higher than T-FNO with ProdLayer (p=2), and the improvement is far from introducing ProdLayer.

Below, we present the results on the DarcyFlow and TorusVisForce dataset for T-FNO.

| $L_{MLP}$               | 2   | 3   | 4   | 5   | 6   | ProdLayer (p=2) |
|-|-|-|-|-|-|-|
| Param (Millon)        | 0.6882 | 0.6893 | 0.6904 | 0.6915 | 0.6925 | 0.6883 |
| DarcyFlow MSE (1e-3)    | 7.500 | 7.951 | 8.630 | 9.373 | 8.165 | 6.919 |
| Few-shot TorusVisForce MSE (1e-2)| 2.104 | 2.097 | 2.142 | 2.145 | 2.180 | 1.757 |















