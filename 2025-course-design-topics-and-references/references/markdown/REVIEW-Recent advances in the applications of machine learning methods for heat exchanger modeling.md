# Recent advances in the
# applications of machine learning
# methods for heat exchanger
# modeling — a review
Junjia Zou 1 , 2 , Tomoki Hirokawa 3 , Jiabao An 1 , Long Huang 1 * and
Joseph Camm 2 *
1 School of Intelligent Manufacturing Ecosystem, Xi ’ an Jiaotong-Liverpool University, Suzhou, China,
2 School of Engineering, University of Liverpool, Brownlow Hill, Liverpool, United Kingdom, 3 Department
of Mechanical Engineering, University of Hyogo, Himeji, Hyogo, Japan
Heat exchanger modeling has been widely employed in recent years for
performance
calculation,
design
optimizations,
real-time
simulations
for
control analysis, as well as transient performance predictions. Among these
applications, the model ’ s computational speed and robustness are of great
interest, particularly for the purpose of optimization studies. Machine learning
models built upon experimental or numerical data can contribute to improving the
state-of-the-art simulation approaches, provided careful consideration is given to
algorithm selection and implementation, to the quality of the database, and to the
input parameters and variables. This comprehensive review covers machine
learning methods applied to heat exchanger applications in the last 8 years.
The reviews are generally categorized based on the types of heat exchangers
and also consider common factors of concern, such as fouling, thermodynamic
properties, and ﬂ ow regimes. In addition, the limitations of machine learning
methods for heat exchanger modeling and potential solutions are discussed,
along with an analysis of emerging trends. As a regression classi ﬁ cation tool,
machine learning is an attractive data-driven method to estimate heat exchanger
parameters, showing a promising prediction capability. Based on this review
article,
researchers
can
choose
appropriate
models
for
analyzing
and
improving heat exchanger modeling.
KEYWORDS
machine
learning,
heat
exchanger,
modeling,
design
optimization,
performance
prediction
### 1 Introduction
A heat exchanger is a device that facilitates heat transfer between ﬂ uids at different
temperatures. It is widely employed in applications such as air-conditioning, refrigeration,
power plants, oil re ﬁ neries, petrochemical plants, natural-gas processing, chemical plants,
sewage treatment, and many others ( Hall, 2012 ; Singh et al., 2022 ). Theoretical analysis,
analytical models, experimental methods, and numerical methods were conventionally
applied to study the heat transfer and ﬂ uid ﬂ ow processes within the heat exchangers
( Du et al., 2023 ). The analytical models generally involve several assumptions in the
derivation of relevant equations and formulae. The process of heat transfer can be
evaluated through classical methods, such as the logarithmic mean enthalpy difference
(LMHD), the logarithmic mean temperature difference (LMTD), ε – NTU, etc. ( Hassan et al.,
OPEN ACCESS
EDITED BY
Baomin Dai,
Tianjin University of Commerce, China
REVIEWED BY
Houpei Li,
Hunan University, China
Hongtao Qiao,
Mitsubishi Electric Research Laboratories
(MERL), United States
Qi Chen,
Xi ’ an Jiaotong University, China
*CORRESPONDENCE
Long Huang,
long.huang@xjtlu.edu.cn
Joseph Camm,
joseph.camm@liverpool.ac.uk
RECEIVED 14 September 2023
ACCEPTED 17 October 2023
PUBLISHED 14 November 2023
CITATION
Zou J, Hirokawa T, An J, Huang L and
Camm J (2023), Recent advances in the
applications of machine learning
methods for heat exchanger modeling — a
review .
Front. Energy Res. 11:1294531.
doi: 10.3389/fenrg.2023.1294531
COPYRIGHT
© 2023 Zou, Hirokawa, An, Huang and
Camm. This is an open-access article
distributed under the terms of the
Creative Commons Attribution License
(CC BY) . The use, distribution or
reproduction in other forums is
permitted, provided the original author(s)
and the copyright owner(s) are credited
and that the original publication in this
journal is cited, in accordance with
accepted academic practice. No use,
distribution or reproduction is permitted
which does not comply with these terms.
Frontiers in Energy Research
frontiersin.org
01
TYPE Review
PUBLISHED 14 November 2023
DOI 10.3389/fenrg.2023.1294531

---

2016 ). However, these techniques are generally based on certain
assumptions and conditions, such as constant physical properties,
steady-state operation, negligible wall heat conduction, uniform
distribution
of
ﬂ ow
properties,
and
a
consistent
air
ﬂ uid
temperature along the ﬁ n height.
For numerical modeling of heat exchangers, discretization of the
refrigerant ﬂ ow ﬁ eld and of the governing equations is required
( Prithiviraj and Andrews, 1998 ). To achieve detailed analysis in
computational solutions, one might consider employing advanced
numerical techniques such as the Finite Volume Method (FVM) or
the Finite Element Method (FEM). It is essential to achieve a balance
of
heat
and
mass
in
each
cell
( Moukalled
et
al.,
2016 ).
Computational Fluid Dynamics (CFD) can be a useful tool in
designing,
troubleshooting,
and
optimizing
heat
exchanger
systems ( Bhutta et al., 2012 ). It transforms the integral and
differential terms in the governing ﬂ uid mechanics equations into
discrete algebraic forms, thereby generating a system of algebraic
equations. These discrete equations are then solved via a computer
to obtain numerical solutions at speci ﬁ c time/space points.
Nonetheless,
numerical
methods
like
CFD
often
require
signi ﬁ cant computational resources ( Thibault and Grandjean,
1991 ; Yang, 2008 ).
In bridging the gap between computational ef ﬁ ciency and
accuracy, several studies on machine learning methods for heat
exchanger analysis have been developed to predict the performance
of heat exchangers. Some representative machine learning methods
recently used to analyze heat exchangers include Arti ﬁ cial Neural
Networks (ANN), Support Vector Machine (SVM), Tree models,
etc., which were shown to generate satisfactory results ( Patil et al.,
2017 ; Zhang et al., 2019 ; Ahmadi et al., 2021 ; Wang and Wang, 2021 ;
Ewim et al., 2021 ; Fawaz et al., 2022 ). An analysis of the number of
papers in this ﬁ eld clearly shows a signi ﬁ cantly growing trend in
recent years, as illustrated in Figure 1 .
In this review, we mainly focus on the review of machine
learning models for air-cooled heat exchangers ( ﬁ nned tube heat
exchangers, microchannel heat exchangers, etc.) in the ﬁ eld of
refrigeration and air-conditioning. The three main objectives of
this paper are: 1) to summarize the studies on machine learning
methods related to heat exchanger thermal analysis over the last
8 years; 2) to compare different machine learning methods
employed in heat exchanger thermal analysis 3) to point out the
limitations and emerging applications of machine learning in heat
exchanger thermal analysis. The organization of this paper consists
of the following ﬁ ve sections: Section 2 summarizes and classi ﬁ es the
machine learning methods. Section 3 summarizes applications of
machine learning methods for modeling heat exchangers in recent
years. Sections 4 , 5 discuss the limitations of ANN modeling for heat
exchanger analysis and future trends in this area, respectively.
### 2 Introduction to machine learning
### models
As illustrated in Figure 2 , the machine learning approaches to
modeling heat exchangers reviewed in this paper include Random
Vector Functional Link Network (RVFL), Support Vector Machine
(SVM), K-Nearest Neighbor (KNN), Gaussian Process Regression
(GPR), Sequential Minimal Optimization (SMO), Radial Basis
Function (RBF), Hybrid Radial Basis Function (HRBF), Least
Square
Fitting
Method
(LSFM),
Arti ﬁ cial
Neural
Networks
(ANN), Random Forest, AdaBoost, Extreme Gradient Boosting
(XGBoost),
LightGBM,
Gradient
Boosting
Tree
(GBT)
and
Correlated-Informed Neural Networks (CoINN). The following
section of the paper focuses on the classi ﬁ cations of the various
methods.
2.1 Classi ﬁ cation of machine learning
methods
Machine learning methods were introduced to predict or regress
the performance indicators of heat exchangers, such as the Nusselt
number ( Nu ), the Heat Transfer Coef ﬁ cient ( HTC ), the pressure
drop ( Δ P ), etc. Based on the learning approaches, machine learning
FIGURE 1
Numbers of publications associated with heat exchangers and machine learning from 2015 till September 2023 (based on data from ScienceDirect).
Frontiers in Energy Research
frontiersin.org
02
Zou et al.
10.3389/fenrg.2023.1294531

---

categories generally include supervised learning, unsupervised
learning, and reinforcement learning. Semi-supervised learning
and active learning were also employed in studies as a machine
learning category, as shown in Figure 2 .
In heat exchanger applications, machine learning techniques
primarily use supervised learning, which involves developing
predictive models using labeled data. Labeled data establishes a
connection between input and output, enabling the prediction
model to generate corresponding outputs for speci ﬁ c inputs.
The essence of supervised learning lies in understanding the
statistical principles that govern the mapping of inputs to
outputs ( Cunningham et al., 2008 ). Unsupervised learning is a
machine learning approach in which predictive models are
developed without relying on labeled data or a clear purpose
( Celebi and Aydin, 2016 ).
Reinforcement learning refers to learning optimal behavior
strategies by an intelligent system in continuous interaction with
the environment ( Wiering and Van Otterlo, 2012 ). For instance,
Keramati,
Hamdullahpur,
and
Barzegari
introduced
deep
reinforcement learning for heat exchanger shape optimization
(Keramati, Hamdullahpur, and Barzegari 2022).
Semi-supervised learning refers to the learning prediction
considering both labeled data sets and unlabeled data ( Zhu and
Goldberg, 2009 ). There is typically a small amount of labeled data
and a large amount of unlabeled data because constructing labeled
data often requires labor and high cost, and the collection of
unlabeled data does not require much cost. This approach aims
to use the information in unlabeled data to assist in labeling data for
supervised learning and achieve enhanced learning results at a lower
cost ( Zhu, 2005 ). Active learning refers to a specialized training
approach where the model actively selects the data it wants to learn
from. Unlike traditional machine learning methods where all
training data is provided upfront, active learning allows the
model to selectively acquire new labeled data during its learning
process. As a result, semi-supervised learning and active learning are
closer to supervised learning. The differences between active
learning and semi-supervised learning are: In active learning, the
algorithm selectively picks the most informative instances for
manual annotation, aiming to enhance model accuracy while
minimizing labeling workload. In contrast, in semi-supervised
learning, the emphasis is not on actively selecting instances.
Instead, it leverages a combination of labeled and unlabeled data
to enhance model generalization and performance through the
integration of these data sources. Chen et al. ( Chen et al., 2021 )
introduced a hybrid modeling method combining the mechanism
with semi-supervised learning for temperature prediction in a roller
hearth kiln, which implies the possibility of being employed in heat
transfer.
2.2 Introduction of the various machine
learning methods
As shown in Table 1 , the classi ﬁ cation of the machine learning
methods considered here is the following:
• Neural networks refer to a series of methods that simulate the
human brain in the hierarchical structure of neurons to
recognize the relationships among speci ﬁ ed data ( Mahesh,
2018 ). Supervised neural network refers to a network
consisting of many neurons connected by weighted links,
which
was
ﬁ rst
introduced
by
Hop ﬁ eld
in
1982
in
biological research ( Hop ﬁ eld, 1982 ; Mahesh, 2018 ). In the
literature, the methods presented for heat exchangers, such as
ANN or RVFL, are distinctive due to the structure of their
network.
• A tree model in machine learning is a type of predictive modeling
tool, transitioning from observed attributes of an entity,
symbolized by the branches, to deductions about the entity ’ s
target value, encapsulated in the leaves ( Clark and Pregibon,
2017 ). This model employs a hierarchical structure to parse the
data, whereby each internal node corresponds to a speci ﬁ c
attribute, each branch signi ﬁ es a decision rule, and each leaf
node represents an outcome or a prediction. The process
FIGURE 2
The classi ﬁ cation of machine learning models.
Frontiers in Energy Research
frontiersin.org
03
Zou et al.
10.3389/fenrg.2023.1294531

---

initiates from the root node and progressively branches out based
on de ﬁ ned decision rules, effectively segmenting the data space
into non-overlapping regions ( Rattan et al., 2022 ). The tree model
de ﬁ nes how to get a prediction score. It can be employed for
classi ﬁ cation and regression, such as Random Forest, AdaBoost,
XGBoost, LightGBM, GBT, etc.
• A Support Vector Machine (SVM) is a widely used supervised
learning model in machine learning. It is used for both
classi ﬁ cation
and
regression
tasks
( Mahesh,
2018 ).
However, it is primarily used in classi ﬁ cation problems.
The basic idea behind SVM is to ﬁ nd a hyperplane in
N -dimensional space (where N is the number of features)
that
distinctly
classi ﬁ es
the
data
points.
The
chosen
hyperplane is systematically optimized to maximize the
“ margin, ” which is de ﬁ ned as the distance to the nearest
data points across different classes. This maximization
strategy is intended to minimize the model ’ s generalization
error, thus enhancing its predictive accuracy for classifying
new instances. In the domain of SVM, Sequential Minimal
Optimization (SMO) serves as an ef ﬁ cient algorithm for
training. SMO can be optimized as a heuristic algorithm
whose basic idea is to optimize only two variables at one
iteration while ﬁ xing the remaining variables ( Sun et al., 2008 ).
• Bayesian regression is a statistical method that uses the
principles of Bayesian statistics to estimate the parameters
of a regression model. It is an alternative to traditional
regression models like linear regression, and it takes a
fundamentally different approach to model the relationship
between the dependent and independent variables ( Sun et al.,
2008 ). In Bayesian statistics, probabilities are treated as a
measure of belief or uncertainty, which can be updated
based on new data. This is especially useful for modeling
systems where uncertainty is inherent, providing a ﬂ exible
framework that allows for iterative re ﬁ nement as new data is
incorporated. Thus, Bayesian regression offers an alternative
but robust way of tackling regression problems
• K-nearest neighbor can solve the classi ﬁ cation and regression
issues related to heat exchangers. A similarity metric is established
within the data space, enabling the prediction of data labels by
utilizing the nearest neighbors in the data space for reference
( Kramer, 2013 ). In the K-nearest neighbor (KNN) algorithm,
when one seeks to predict the label of an unobserved data point,
the algorithm speci ﬁ cally identi ﬁ es ‘ K ’ instances from the training
set that are in closest proximity to the given point. The
determination of “ proximity ” is generally quanti ﬁ ed using a
distance metric, with the Euclidean distance being the most
commonly employed metric in numerous applications. For
illustrative purposes, if we set K to 3, the KNN procedure will
focus on the three most proximate training data instances relative
to the unobserved point to facilitate the prediction. In the realm of
classi ﬁ cation, the predominant label amongst these three
neighbors is then allocated to the unobserved data point. In
the context of regression analysis, the algorithm might predict the
label by computing the mean value from the labels of the three
nearest neighbors.
### 3 Machine learning models applied for
### heat exchanger modeling
Traditional physics-based models may encounter dif ﬁ culties when
dealing with complex and non-linear problems, requiring extensive
specialist knowledge and experience. In this context, machine learning
methods have been introduced to the ﬁ eld of heat exchangers. Machine
learning models built upon experimental or numerical data can
improve state-of-the-art simulation methodologies. Machine learning
can reduce calculation time, increase prediction accuracy, and handle
complex and non-linear issues. In recent years, there have been notable
advances in the application of machine learning methods in the ﬁ eld of
heat exchangers, such as using machine learning to predict heat transfer
coef ﬁ cients ( Section 3.2.1 ), pressure drop ( Section 3.2.2 ), and heat
exchanger performance ( Section 3.2.3 ) performing real-time analysis
of complex experimental data, and optimizing large-scale thermal
systems.
This section reviews the recent advances in applications of
machine learning methods for heat exchanger modeling in the
following categories: ( Section 3.2.1 ) Modeling of Heat Transfer
Coef ﬁ cient (HTC), ( Section 3.2.2 ) Modeling of pressure drops,
( Section 3.2.3 ) Modeling of heat exchanger performance ( Section
3.3 ) Fouling factors, ( Section 3.4 ) Refrigerant thermodynamic
properties, and ( Section 3.5 ) Flow pattern recognition based on
machine learning methods.
TABLE 1 The primary classi ﬁ cation of machine learning.
Machine learning
categories
Output type
Learning strategy
Methods
Supervised Neural network
Classi ﬁ cation,
Regression
Minimize the loss function
Arti ﬁ cial Neural Networks (ANN), Random Vector Functional Link
Network (RVFL), Correlated-Informed Neural Networks (CoINN),
Hybrid Radial Basis Function (HRBF), Radial Basis Function (RBF)
Tree model
Classi ﬁ cation,
Regression
Maximum likelihood estimation for
regularization
Random Forest (RF), AdaBoost, Extreme Gradient Boosting (XGBoost),
LightGBM, Gradient Boosting Tree (GBT)
Support vector machine
Binary
Classi ﬁ cation
Soft margin maximization
Sequential Minimal Optimization (SMO)
Bayesian
Classi ﬁ cation
Maximum likelihood estimation,
maximum a posteriori estimation
Gaussian Process Regression (GPR)
K-nearest neighbor
Classi ﬁ cation,
Regression
Minimization of distance
K-Nearest Neighbor (KNN)
Frontiers in Energy Research
frontiersin.org
04
Zou et al.
10.3389/fenrg.2023.1294531

---

3.1 Heat exchangers
For the heat exchanger reviewed in this section, as shown in
Figure 3A , microchannel heat exchangers consist of small-scale ﬁ nned
channels etched in silicon wafers and a manifold system that forces a
liquid ﬂ ow between ﬁ ns ( Harpole and Eninger, 1991 ). As shown in
Figure 3B , the shell and tube heat exchangers are devices consisting of
a vessel containing either a bundle of multiple tubes or a single tube
bent several times, with the wall of the tube bundle enclosed in the
shell being the heat transfer surface. This design has the advantages of
simple structure, low cost, and wide ﬂ ow cross-section ( Mirzaei et al.,
2017 ). As shown in Figure 3C , a plate heat exchanger is more compact
than the shell and tube heat exchanger design because of its smaller
volume and larger surface area and because its modular design can
increase or reduce the number of required plates to satisfy different
requirements, retaining excellent heat transfer characteristics ( Abu-
Khader, 2012 ). As shown in Figure 3D , Tube-Fin Heat Exchangers
(TFHXs) are important components in heat pump and air
conditioning systems, which consists of a bundle of ﬁ nned tubes
( Li et al., 2019 ).
3.2 Parameters modeling of heat exchangers
This subsection summarizes the use of machine learning in modeling
heat exchangers, with each subsubsection describing different parameters
predicted in the research, providing a comprehensive summary of the
classi ﬁ cations. The different types of heat exchangers are shown in Tables
2 – 6 . Tables 2 – 6 delineate speci ﬁ c literature references pertaining to each
unique type of heat exchanger. Each table incorporates speci ﬁ c references
correlating to a distinct type of heat exchanger. This systematic
organization of information aims to streamline the process to
effectively locate and review pertinent literature based on the unique
type of heat exchanger they are researching.
To conduct a robust, quantitative assessment of the models
introduced in this research, we have incorporated a range of error
metrics, as cataloged in Tables 2 – 9 . These metrics not only facilitate
an empirical evaluation of model performance but also provide
prospective users with a criteria-based framework for model
selection relative to speci ﬁ c applications. We delineate the
mathematical equations that form the basis for each type of error
metric employed as shown in the following equations.
Mean Relative Error (MRE) is a metric that quanti ﬁ es the
relative size of the prediction errors with respect to the actual
observed values. The formula for MRE is:
MRE  1
n 
n
i  1
| X predict − X real |
X real
(1 )
Mean Absolute Error (MAE) is a popular metric for regression.
It measures the average absolute difference between observed and
predicted values. The formula is:
MAE  1
n 
n
i  1 | X predict − X real |
(2 )
FIGURE 3
(A) Microchannel heat exchangers. (B) Shell and tube heat exchangers adapted from ( Foley, 2013 ). (C) Plate heat exchangers. (D) Tube- ﬁ n heat
exchangers.
Frontiers in Energy Research
frontiersin.org
05
Zou et al.
10.3389/fenrg.2023.1294531

---

Root Mean Squared Error (RMSE) is another commonly
used regression metric. It ﬁ rst calculates the square of the
difference between each observed value and its predicted
value, averages these, and then takes the square root. Its
formula is:
RMSE  1
n


n
i  1 X predict − X real


2

(3 )
Median Absolute Error (MedAE) is similar to MAE, but instead of
using the mean of the absolute errors, it uses the median. The formula is:
MedAE  1
n 
n
i  1 | X predict − X median |
(4 )
R-squared, also known as the coef ﬁ cient of determination,
is used to measure how well the model explains the variability
TABLE 2 Machine learning applications for modeling microchannel heat exchangers.
Authors
Type of machine
learning
Input
Output
Error analysis
Moradkhania et al. ( Moradkhani
et al., 2022a )
GPR
Pr tp , Re tp , x , P red , R , We go , Fr l , Bo
Nusselt number
MRE 4.50%
RBF
MRE 19.41%
HRBF
MRE 24.53%
Ma et al. (Y. Ma et al., 2022 )
GBT
α , N , Re and A
Nusselt number
RMSE 219.74% R 2
99.90%
Pumping power
RMSE 5.4% R 2
99.95%
Hughes et al. ( Hughes et al., 2022 )
SVR
Re g , Re f , Bo , We , ScV , Scl , Pr f , Ja , T *
Nusselt number
MAE 4.95%
for SVR
RFR
MAE 8.6% for RFR
GB
MAE 6.2% for GB
ANN
MAE 5.3%
for ANN
Re g , Re f , Bd , We , ScV , ScL
Friction factor
MAE 5.0% for SVR
MAE 8.9% for RFR
MAE 7.0% for GB
MAE 5.0%
for ANN
Zhou et al. ( Zhou et al., 2020a )
ANN
Bd , Co , Fr f , Fr fo , Fr g , Fr go , Ga , Ka , Pr f , Pr g , Re f , Re fo , Re g ,
Re go , Su f , Su g , Su fo , Su go , We f , We fo , We g , We go
Heat transfer
coef ﬁ cient
MAE 6.80% R 2 98%
Random Forest
MAE 18.56%
R 2 87%
AdaBoost
MAE 34.60%
R 2 75%
XGBoost
MAE 9.06% R 2 97%
Montanez-Barrera et al.
( Montañez-Barrera et al., 2022 )
CoINN
Mixture vapor quality, Micro-channel inner diameter,
Available pressure drop correlation
Pressure drop
MRE 6%
Qiu et al. ( Qiu et al., 2021 )
ANN
Bd , Bo , Fr f , Fr fo , Fr g , Fr go , Fr tp , Pr f , Pr g , Pe g , Pe f , Re f ,
Re fo , Re g , Re go , Re eq , Su f , Su g , We f , We fo , We g , We go , We tp
Flow boiling pressure
drop
MAE 9.58%
for ANN
KNN
MAE 10.38%
for KNN
XGBoost
MAE 13.52% for
XGBoost
Light GBM
MAE 14.49% for
Light GBM
Zhu et al. ( Zhu et al., 2021a )
ANN
The channel geometry size, Fluid thermal properties, The
working ﬂ uid conditions and Heat ﬂ ux or other derived
dimensionless parameters
HTC (boiling and
condensation)
MRE 11.41% for
boiling
MRE 6.06% for
condensation
Frontiers in Energy Research
frontiersin.org
06
Zou et al.
10.3389/fenrg.2023.1294531

---

TABLE 3 Machine learning applications for modeling plate heat exchangers.
Authors
Type of machine
learning
Input
Output
Error analysis
Amal ﬁ et al. ( Amal ﬁ and
Kim, 2021 )
RF
Mass ﬂ ow rate, Saturation temperature, Heat ﬂ ux, and
Geometrical parameters
Nusselt number
MAE 10.0%
Local frictional pressure
gradient
MAE 10.3%
Longo et al. ( Longo et al.,
2020c )
GBM
Φ , β / β max , Prf , Speci ﬁ c kinetic energy number, P / Pc ,
and Boiling or condensation
Frictional pressure
gradient
MAE of 6.6%
Longo et al. ( Longo et al.,
2020b )
ANN
Φ , β / β max , Pr f , Re eq , and Reduced pressure
Heat transfer factor
(boiling)
MAE 4.8%
Longo et al. ( Longo et al.,
2020a )
ANN
Δ T , Δ T sup , Φ , Re eq , Pr f
Heat transfer factor
(condensation)
MAE 3.6%
Gupta et al. ( Gupta et al.,
2017 )
ANN
Q , P 1, P 2, Pcd , Phd , T 1, T 3
Outlet cold ﬂ uid
temperature
Average error of 0.25%
for ANN
ANFIS
Average error of 0.896%
for ASNFIS
Outlet hot ﬂ uid
temperature
Average error of 0.19%
for ANN
Average error of 0.192%
for ASNFIS
TABLE 4 Machine learning applications for modeling tube- ﬁ n heat exchanger.
Authors
Type of
machine
learning
Input
Output
Error
analysis
Naja ﬁ et al. ( Naja ﬁ et al.,
2021 )
RF
xv , Re , ( 1 − x ) / x , Re f , Re go ,
Frictional pressure drop
MARD 6.72%
Ardam et al. ( Ardam
et al., 2021 )
RF
X , F go , n , Bo , e / D int
Pressure drop
MARD 6.41%
Xie et al. ( Xie et al.,
2022a )
ANN
L , α , β
Nusselt number
R 2 99%
Friction factor
R 2 99.8%
Du et al. ( Du et al., 2020 )
ANN
θ , Re , DC , Re w , T ia , T iw , Ntr , Ntp the outer length of the
major axis, the outer length of the minor axis, Dc
Nusselt number
MSE 78.97%
Friction factor
MSE 1.08%
Skrypnik et al.
( Skrypnik et al., 2022 )
ANN
Re , P / D , Helical fin height / D , θ /90 Inter- ﬁ n distance/
Helical ﬁ n height, Number of helical ﬁ n starts
Nusselt number
MAE 16.3%
Friction factor
MAE 11.8%
Subbappa et al.
( Subbappa et al., 2022 )
ANN
Refrigerant, Tubes per bank, Tubes per bank per circuit
(i.e., circuitry), Tube banks, Tube length, Fins per inch, Air
velocity, Refrigerant temperature, Refrigerant G
Heat transfer and Refrigerant pressure
drop
ANN and SVR ±
20% for 90%
RR
SVR
Li ( Li et al., 2016 )
RSM based NN
T s,in , x in , m r , V a , T db,in , and T wb,in
Total cooling capacity, Sensible heat
ratio, and Pressure drops on both
refrigerant and air sides
Dry
condition
R 2 > 99.8%
Wet
condition
R 2 > 97.4%
Krishnayatra
( Krishnayatra et al.,
2020 )
KNN
Fin spacing, Fin thickness, Material, and Convective heat
transfer coef ﬁ cient
Overall ef ﬁ ciency
R 2 90.14% k = 2
Total effectiveness
R 2 85.37% k = 8
Frontiers in Energy Research
frontiersin.org
07
Zou et al.
10.3389/fenrg.2023.1294531

---

among the observed values. It ranges between 0 and 1, with
values closer to 1 indicating a better ﬁ t. Its formula is:
R 2  1 −
 n
i  1 X predict − X real


2
 n
i  1 X predict − X average


2
(5 )
3.2.1 Modeling of Heat Transfer Coef ﬁ cient
The Heat Transfer Coef ﬁ cient (HTC) plays a pivotal role in the
design and optimization of heat exchangers. It is a key parameter
that describes the rate of heat transfer per unit area, per unit of
temperature difference. In ﬂ uid dynamics and heat transfer studies,
TABLE 5 Machine learning applications for modeling shell and tube heat exchanger.
Authors
Type of machine
learning
Input
Output
Error analysis
El-Said et al. ( El-Said et al., 2021 )
RVFL
Cold ﬂ uid, and injected air volume ﬂ ow rates
Outlet temperature of
cold ﬂ uids
RMSE 52.78% for
RVFL
SMO
RMSE 149.6%
for SMO
SVM
RMSE 53.56%
for SVM
KNN
RMSE 140.0%
for KNN
Outlet temperature of
hot ﬂ uids
RMSE 71.91% for
RVFL
RMSE 247.7%
for SMO
RMSE 174.1%
for SVM
RMSE 185.5%
for KNN
Pressure drop
RMSE 0.9093% for
RVFL
RMSE 2.3525%
for SMO
RMSE 1.5391%
for SVM
RMSE 0.8944%
for KNN
Kunjuraman and Velusamy ( Kunjuraman
and Velusamy, 2021 )
ANN
CF , FIT , SF
Condensate
temperature
MRE 0.971%
for ANN
ANFIS
RMSE1.175%
for ANN
R 2 94.56% for ANN
MRE 0.381% for
ANFIS
RMSE 0.532% for
ANFIS
R 2 99.98% for
ANFIS
Roy and Majumder ( Roy and Majumder,
2019 )
FFBN
Tube con ﬁ gurations (30, 40, 60, and 90), Different
ﬂ uids, surface, Temperature
Exergetic Plant
Ef ﬁ ciency
Accuracy 98.11%
Energetic Cycle
Accuracy 97.40%
Ef ﬁ ciency
Accuracy 96.35%
Electrical Power Cost
Accuracy 97.23%
Fouling factor
Accuracy 98.32%
Muthukrishnan et al. ( Muthukrishnan
et al., 2020 )
SVM
Nt , Sb , Nb , Dc
Heat transfer rate
Accuracy > 90%
Frontiers in Energy Research
frontiersin.org
08
Zou et al.
10.3389/fenrg.2023.1294531

---

TABLE 6 Machine learning applications for modeling other heat exchangers.
Authors
Type of heat
exchangers
Type of
machine
learning
Input
Output
Error analysis
Azizi and Ahmadloo ( Azizi
and Ahmadloo, 2016 )
Inclined tube
ANN
IA , G , T s , xv
Heat transfer coef ﬁ cient
MAE 1.94%
R 2 99.5%
Zheng et al. ( Zheng et al.,
2022 )
Heat exchange channels
with bulges
GRNN
Order of the bulge heights at
different locations (6 nodes)
Heat transfer coef ﬁ cient
Both R 2 > 97% for
GRNN and RF
RF
Moradkhani et al.
( Moradkhani et al., 2022b )
Inside smooth helically
coiled tubes
GPR
Re tp , Pr tp , X , P red , Bo , Dt / Dc
and Frf
Boiling heat transfer
coef ﬁ cient
MRE 5.93% for GPR
RBF
MRE 6.67% for RBF
MLP
MRE 9.27% for MLP
Kwon et al. ( Kwon et al.,
2020 )
Rough cooling channel
RF
Height of channel geometries
e1, e2, e3, e4, e5 (5 nodes)
Convection heat transfer
coef ﬁ cients
R 2 > 96.6%
Dalkilic et al. ( Dalkilic et al.,
2019 )
Smooth pipe
ANN
Re , Gr Δ T *10 − 6 6, Pr , Bd , f ,
f 0, μ w / μ b , fvp
Tube length averaged the
Nusselt number and Nusselt
number in forced convection
Accuracy ±5%
Chokphoemphun et al.
( Chokphoemphun et al.,
2020 )
Grooved channel air heater
NN
Turbulator, Depth ratio, IA ,
and Re
Nusselt number
R 2 99.8864%
Friction factor
R 2 99.9772%
Thermal enhancement factor
R 2 99.8858%
Alireza Zendehboudi*,
Xianting Li ( Zendehboudi
and Li, 2017 )
Inclined smooth tubes
PSO-ANN
IA , G , xv , and T s
Pressure drop
R 2 96.092% for
PSO-ANN
GA-LSSVM
R 2 99.931% for GA-
LSSVM
Hybrid-ANFIS
R 2 99.932% for
Hybrid-ANFIS
GA-PLCIS
R 2 99.937% for GA-
PLCIS
MSE RRMSE et al
Frictional pressure drop
R 2 97.753% for
PSO-ANN
R 2 99.932% for GA-
LSSVM
R 2 99.940% for
Hybrid-ANFIS
R 2 99.944% for GA-
PLCIS
MSE RRMSE et al
Garcia et al. ( Garcia et al.,
2018 )
R407C in horizontal tubes
ANN
Dt , G , P s , and xv
Pressure drop
MAE 6.11%
R 2 99.9%
Shojaeefard et al. ( Shojaeefard
et al., 2017 )
Compact heat exchanger
(evaporator)
Numerical
T a,in,db , T a,in,wb , P ref,in , T ref,in ,
_ m ref , and Va
Q evap , P ref,out , T ref,out ,
T a,out,db , and T a,out,wb
RMSE (avg) 228.6%
for Numerical
FFNN
RMSE (avg) 501.7%
for FFNN
GA-trained
RMSE (avg) 479.1%
for GA-trained
RNN
RMSE (avg) 116.9%
for RNN
MSE, R 2
(Continued on following page)
Frontiers in Energy Research
frontiersin.org
09
Zou et al.
10.3389/fenrg.2023.1294531

---

TABLE 6 ( Continued ) Machine learning applications for modeling other heat exchangers.
Authors
Type of heat
exchangers
Type of
machine
learning
Input
Output
Error analysis
Uguz and Iperk ( Uguz and
Ipek, 2022 )
Compact heat exchanger
ANN
T hw , T hwi , T cw , T cwi , and m _
cw
T cw,out
R 2 96.0% for ANN
MLR
R 2 96.1% for MLR
SVR
R 2 94.2% for SVR
T hw,out
MSE, MAE and
MedAE
Peng and Ling et al. ( Peng and
Ling, 2015 )
Compact heat exchangers
SVR
Fin height, Fin pitch, Fin,
Thickness, Fin length, and
Reynolds number at the air
side
Friction factor
MSE 2.645 *10-4
Colburn factor
MSE 1.231 * 10-3
Azizi et al. ( Azizi et al., 2016 )
Gas – liquid ﬂ ow in
horizontal, upward and
downward inclined pipes
ANN
IA , Re sg , and Re sl
Void fraction
MAE 1.52%
R 2 99.48%
Bhattacharya et al.
( Bhattacharya et al., 2022 )
Heat exchanger
CNN-GRU SSM
_ m a in , P atm , T in , RH in , _ m ref in ,
h in , P out , h out
p inlet , p outlet , h inlet , h outlet ,
Q total , _ m total
Maximum
percentage error is
capped at 0.2%
Li et al. ( Li et al., 2023 )
Printed circuit heat
exchangers
ANN
_ m , D , T in , P in , Wall heat ﬂ ux,
V in , ρ in , Length, Re and Pr
HTC
R 2 99.94% for ANN
XGBoost
LightGBM
Δ P
R 2 99.96% for ANN
Random forest
Chen et al. ( Chen et al., 2023 )
Energy pile heat pump
system
ANN
T amb , RH amb , T room ,
RH room , P sys
Coef ﬁ cient of performance
MAE 31.4%
TABLE 7 Prediction of the fouling factor for heat exchangers based on machine learning methods.
Authors
Type of heat
exchangers
Type of
machine
learning
Inputs
Outputs
Error analysis
Hosseini et al.
( Hosseini et al., 2022 )
Preheat exchanger networks
of petroleum re ﬁ neries
GPR
Operation time, Surface temperature, Fluid
velocity, Fluid density, Fluid temperature, and
Equivalent diameter
Fouling factor
(m 2 K/kW)
R 2 13.89%
for GPR
DT
R 2 16.64% for DT
Bagged Trees
R 2 8.86% for
Bagged Trees
SVR
R 2 35.39% for SVR
Mohanty ( Mohanty,
2017 )
Shell and tube heat
exchanger
ANN
The Inlet temperature, Re, and Mass ﬂ ow rate
on both tube and shell sides
Tube-side temperature
difference,
3.1 (predicted
value)
Shell side temperature
difference
2.2 (predicted
value)
Ef ﬁ ciency
7.26% (predicted
value)
Kuzucanl ı et al.
( Kuzucanl ı et al.,
2022 )
Plate heat exchange
Naïve Bayes
Varied ﬂ ow rate, Inlet temperatures
Heat transfer
coef ﬁ cient and fouling
factor
100% for Naïve
Bayes
DT
99.3% for decision
tree
KNN
96.3% for KNN
(Predict accuracy)
Sundar et al. ( Sundar
et al., 2020 )
Cross- ﬂ ow heat exchanger
Deep learning
T fin , T win , _ m w , f / _ m w , _ m flue , f / _ m flue ,
T fo , T wo ,
Overall fouling factor
R 2 99.86%
Frontiers in Energy Research
frontiersin.org
10
Zou et al.
10.3389/fenrg.2023.1294531

---

the Nusselt number is often introduced as a dimensionless
parameter delineating the relative signi ﬁ cance of convective heat
transfer to conductive heat transfer across a de ﬁ ned boundary. It
essentially offers a normalized representation of the Heat Transfer
Coef ﬁ cient (HTC). Accurate prediction of HTC can lead to more
ef ﬁ cient design and optimization of heat exchangers, resulting in
improved performance and reduced energy consumption ( Zhu et al.,
2021 ). This subsection summarizes and categorizes studies related to
the prediction of the Heat Transfer Coef ﬁ cient (HTC) presented in
recent literature. The classi ﬁ cation is primarily based on the types of
input parameters used, with a particular focus on distinguishing
between dimensionless parameters and structural parameters.
Additionally, a separate classi ﬁ cation is conducted based on the
different sources of data used, including historical literature,
experimental data, and Computational Fluid Dynamics (CFD)
simulations.
A plethora of research efforts has been methodically
invested in the predictive modeling of the Heat Transfer
Coef ﬁ cient (HTC), focusing primarily on the in ﬂ uence of
structural parameters to construct effective machine learning
training datasets. For instance, Zheng et al. ( Zheng et al., 2022 )
introduced General Regression Neural Network (GRNN) and
RF algorithms to predict HTC in heat exchange channels with
bulges with the inputs of each bulge height at different
locations. Other works by Moradkhani et al. ( Moradkhani
et al., 2022a ) and Kwon et al. ( Kwon et al., 2020 ) have
delved into the speci ﬁ cs of boiling and convection heat
transfer coef ﬁ cients, respectively. In these works, the effect of
surface roughness on HTC has not been suf ﬁ ciently explored,
and the amount of measurement data on the topic is insuf ﬁ cient
to include the impact of surface roughness in predictive models.
Therefore, the empirical model that incorporates the effects of
surface roughness into the HTC prediction model needs further
research.
For a predictive model, the exclusive reliance on structural
parameters may prove insuf ﬁ cient. Some studies in the
literature have indeed embraced models where the database
inputs consist of dimensionless numbers or physical properties,
which
can
standardize
data,
enhance
the
stability
and
performance of the model, and make the model ’ s output
easier to understand and interpret. For instance, Longo et al.
( Longo et al., 2020b ) developed ANN to estimate the boiling
heat transfer coef ﬁ cients of refrigerants in Brazed Plate Heat
Exchangers (BPHEs), where the inputs are the corrugation
enlargement
ratio
( Φ ),
the
reduced
inclination
angle
( β / β max ), the liquid Prandtl number ( Pr f , ), the equivalent
Reynolds number ( Re eq ), the boiling number ( Bo ), and the
reduced pressure ( P / P cr ).
In understanding the various methodologies applied in machine
learning modeling, a clear distinction arises from the source of
databases
utilized in
various
research.
A portion
of
these
investigations derives data from pre-existing literature, while
TABLE 8 Prediction of the thermodynamic properties for refrigerants based on machine learning methods.
Authors
The refrigerants
Type of
machine
learning
Inputs
Outputs
Error
analysis
Zhi et al. ( Zhi et al., 2018 )
R1234ze(E), R1234yf, R32, R152a, R161 R245fa
ANFIS
T , P , ρ
Viscosity
MAE 414.96%
for ANFIS
RBFNN
MAE 500.57%
for RBFNN
BPNN
MAE 515.61%
for BPNN
R 2 , RMSE
Gao et al. ( Gao et al., 2019 )
HFC-23, HFC-32, HFC-125, HFC-134a, HFC-143a,
HFC-152a, HFC-161, HFC-227ea, HFC-236fa, HFC-
245fa, HFO-1234yf, HFO-1234ze(E)
ANN
P red , 1 − Tr , ω , Pc / Pcr
Reduced residual
heat capacity
MAE 0.779%
RMSE 11.05%
R 2 99.52%
MAD 13.6%
Wang et al. (X. Wang
et al., 2020 )
R125, R134a, R143a, R152a, R161, R227ea, R236fa,
R32, R1234yf, R1234yf, R1234ze(E), R1336mzz(Z)
ANN
P red , Tr , M , ω
Viscosity
MSE 1.019e-5
Thermal
conductivity
MSE
1.46774e-6
Zolfaghari and Youse ﬁ
( Zolfaghari and Youse ﬁ ,
2017 )
HFC-134a, Decane, Octane, Heptane, Diethyl
carbonate, Dimethyl carbonate, n-Nonane,
n-Dodecane, CO2
ANN
T , P , Mole fraction,
Density
MAE 0.34%
Total molecular weight,
Normal boiling
temperature
Nabipour ( Nabipour,
2018 )
R143a-R227ea, R32-R125, R290-152a, R32-R227ea,
R143a-R125, R125-R152a, R32-R134a, R125-R134a,
R134a-R152a, R290-R600a, R290-R32, R134a-
R143a, R290-RE170, R22-R115, R134a-R1234yf,
R134a-R1234ze(E), R32-R1234yf, R32-R1234ze(E)
ANN
T , Pc , Tc , Critical
volume, ω
Surface tension
MRE 0.7582%
R 2 99.97%
Frontiers in Energy Research
frontiersin.org
11
Zou et al.
10.3389/fenrg.2023.1294531

---

some data are procured from Computational Fluid Dynamics
(CFD). Amal ﬁ and Kim ( Amal ﬁ and Kim, 2021 ) introduced the
randomized decision trees to predict the Nu. The consolidated
experimental database was collected from Amal ﬁ et al. ( Amal ﬁ
et al., 2016a ). The results showed that it could signi ﬁ cantly
improve the prediction of the thermal performance of two-phase
cooling systems compared to the study of Amal ﬁ et al. ( Amal ﬁ et al.,
2016b ), which used physics-based modeling methods. Differently,
Ma et al. ( Ma et al., 2022 ) constructed a GBT tree model based on the
output of CFD simulations of microchannel refrigerant ﬂ ow to
predict the Nusselt number and the pumping power ( WPP ).
This study demonstrates that the most in ﬂ uential parameters are
A , N , and α , while Nu shows an insensitivity to the Reynolds
number of the inlet ﬂ ow.
TABLE 9 Prediction of ﬂ ow regime based on machine learning methods.
Authors
Type of experimental
subjects
Type of
machine
learning
Input
Flow regime
Error analysis
Shen et al. ( Shen et al.,
2020 )
Microchannels heat exchanger (Liquid-
liquid biphasic ﬂ ow patterns in the
per ﬂ uoroalkoxy capillary with the inner
diameter of 1 mm)
CNN
32,383 ﬂ ow pattern
images with labeled
classi ﬁ cation
(Camera) Annular/parallel
ﬂ ow, Slug ﬂ ow, Droplet ﬂ ow,
Wavy annular ﬂ ow, and
Dispersed ﬂ ow
Prediction
accuracy > 98%
Ahmad et al. ( Ahmad
et al., 2022 )
Millimetric closed-loop pulsating heat
pipe (PHP)
DL
648 images ﬂ ow pattern
images with labeled
classi ﬁ cation
(Camera) Bubbly ﬂ ow, Slug-
plug ﬂ ow, Elongated ﬂ ow, and
Annular ﬂ ow
Prediction
accuracy 96%
Giri Nandagopal et al.
( Giri Nandagopal and
Selvaraju, 2016 )
Microchannel heat exchangers
ANN-PR
Con ﬂ uence angle,
Super ﬁ cial velocity of
water, Super ﬁ cial velocity
of dodecane
(Camera) Slug Flow, Bubble
ﬂ ow, Annular ﬂ ow, Elongated
slug ﬂ ow, Deformed ﬂ ow,
Strati ﬁ ed ﬂ ow
R 2 83.83% for ANN-PR
ANN-FF
R 2 88.64% for ANN-FF
CFN
R 2 95.34% for CFN
PNN
R 2 97.66% foe PNN
GRNN
R 2 98.8% for GRNN
ANFIS
R 2 77.64% for ANFIS
Giri Nandagopal et al.
( Giri Nandagopal
et al., 2017 )
Microchannel heat exchangers
ANN-PR
Con ﬂ uence angle,
Super ﬁ cial velocity of
water, Super ﬁ cial velocity
of dodecane
(Camera) Slug Flow, Bubble
ﬂ ow, Annular ﬂ ow, Elongated
slug ﬂ ow, Deformed ﬂ ow,
Strati ﬁ ed ﬂ ow
R 2 93.95% for ANN-PR
ANN-FF
R 2 91.98% for ANN-FF
CFN
R 2 96.6% for CFN
PNN
R 2 95.58% for PNN
GRNN
R 2 98.8% for GRNN
ANFIS
R 2 90.22% for ANFIS
Roshani et al.
( Roshani, Nazemi,
and Roshani, 2017 )
A Pyrex-glass pipe with outside
diameter 100 mm, thickness 2.5 mm
and length 50 cm
RBF NN
With two full energy
peaks in both
transmission detectors
(Gamma ray) Annular,
Strati ﬁ ed, Bubbly
MAE 0.6026%
MRE 0.0496%
Hanus et al. ( Hanus
et al., 2018 )
Horizontal pipeline (inner diameter of
30 mm)
PNN
9 feature values of signal
analysis
(Gamma ray) Slug, Plug, Plug-
Bubble, and Bubble
Accuracy = 1 for the
four classi ﬁ cations
unless Single DT
(0.992)
MLP
RBF
SVM
PNN chosen as the best
Single DT
K – means
Giannetti et al.
( Giannetti et al., 2020 )
Microchannel heat exchangers
ANN
Re , Fr , Ca , β
(Prigogine ’ s Theorem
( Onsager, 1931 ; Prigogine and
Van Rysselberghe, 1963 ))
Take-off ratio
RMSE 4.3%
R 2 98.02%
Godfrey Nnabuife
et al. ( Godfrey et al.,
2021 )
S-shaped pipeline
Deep NN
Vectors that contain all
the information
(CWDU) Annular, Churn,
Slug, and Bubbly
Predict accuracy
99.01%
Khan et al. ( Khan
et al., 2022 )
Horizontal pipe with 5 cm inner
diameter
CNN (ResNet
and Shuf ﬂ eNet)
Scalograms convered
from pressure detectors
(Pressure signals) Strati ﬁ ed
ﬂ ow, Slug ﬂ ow, Annular ﬂ ow
ResNet50 85.7%
Shuf ﬂ eNet 82.9%
Frontiers in Energy Research
frontiersin.org
12
Zou et al.
10.3389/fenrg.2023.1294531

---

3.2.2 Modeling of pressure drops
Pressure drop or pressure differential refers to the decrease in
pressure that a ﬂ uid experiences as it ﬂ ows through a conduit, valve,
bend, heat exchanger, or other equipment. This decrease in pressure
is due to factors such as frictional resistance, local resistance, or
thermal effects ( Ardhapurkar and Atrey, 2015 ). It is imperative to
minimize the pressure drop across a heat exchanger (HX) because a
reduced pressure drop directly translates to decreased pumping
power and a subsequent reduction in the energy input required for
the system in which the HX operates. This section summarizes and
categorizes historical literature related to the prediction of pressure
drop. The categorization is primarily based on the type of machine
learning method used, including predictions based on neural
networks, random forest algorithms, predictions support vector
regression,
and
other
methods.
Additionally,
some
studies
speci ﬁ cally focus on predicting frictional pressure drop.
In literature, ANN can be considered one of the most common
machine learning models used for pressure drop prediction.
Montanez-Barrera et al. ( Montañez-Barrera et al., 2022 ) and Qiu
et al. ( Qiu et al., 2021 ) employed ANN or Correlated-informed
neural networks to predict pressure drops. In addition, Qiu et al.
( Qiu et al., 2021 ) also explored other techniques, including XGBoost
and GBM. Subbappa et al. ( Subbappa et al., 2022 ) employed three
different
methods,
Ridge
Regression
(RR),
Support
Vector
Regression (SVR), and ANN. In this work, it is reported that the
radiator, condenser, and evaporator baseline models are developed
with a different database. The inputs involve the refrigerant
properties, the number of tubes per bank, the number of tubes
per bank per circuit (i.e., circuitry), the tube banks, the tube length,
the number of ﬁ ns per inch, the air velocity, the refrigerant
temperature, and the refrigerant mass ﬂ ux. It is concluded that
ANN and SVR can avoid the expensive simulations with a
reasonable error of ±20% for the testing data used in the study.
However, the validation of this study needs to be veri ﬁ ed using high-
ﬁ delity models, which refer to models that are highly accurate and
detailed. It closely represents or mirrors the real-world system or
situation that is being modeled. High- ﬁ delity models aim to capture
the intricacies and complexity of the actual system to the maximum
extent possible ( Jagielski et al., 2020 ). The machine learning models
that have been trained substantially expedite the investigation of the
design space, leading to a considerable reduction in engineering time
required to reach designs that are nearly optimal.
Despite the highlighted prominence of ANN in the realm
of machine learning models,
various other computational
approaches are also employed as evidenced in the literature.
Ardam et al. ( Ardam et al., 2021 ) developed the prediction of
pressure drop based on the Random Forest algorithm in micro-
ﬁ nned tubes with evaporating R134a ﬂ ow. It employed ﬁ ve
features ( X , f go , n , Bo , e / Dint ) selected among 19 features,
which
showed
the
highest
prediction
accuracy
through
parametric optimization. The results showed that the proposed
methodology is better than the physical model used to respresent
the same data ( Shannak, 2008 ). In addition, Zendehboudi and Li
( Zendehboudi and Li, 2017 ) predicted Δ P and the frictional
pressure drop in inclined smooth tubes based on different
models,
such
as
PSO-ANN,
GA-LSSVM,
Hybrid-ANFIS,
and GA-PLCIS. The two databases are collected from the
experimental study of Adelaja et al. ( Adelaja et al., 2017 ).
In the context of pressure drop predictions discussed thus far, it
is of considerable importance to recognize the frictional pressure
drop as a major component contributing to the overall pressure
losses. Some studies have focused on investigating the frictional
pressure drop in heat exchangers, such as Naja ﬁ et al. ( Naja ﬁ et al.,
2021 ), Xie et al. ( Xie et al., 2022 ), Skrypnik et al. ( Skrypnik et al.,
2022 ), Peng and Ling et al. ( Peng and Xiang, 2015 ) and Du et al. (X.
Du et al., 2020 ), introduced the estimation model of the friction
factor using different machine learning methods. Naja ﬁ et al. ( Naja ﬁ
et al., 2021 ) demonstrated that data-driven estimation of frictional
pressure drop provides greater prediction accuracy compared to
theoretical physical models ( Chisholm, 1967 ) for two-phase
adiabatic air-water ﬂ ow in micro- ﬁ nned tubes using the Random
Forest model. Their research focused on ﬁ ve dimensionless features
( xv , Re , ( 1 − x ) / x , Re f , Re go ) selected from 23 features which are
slightly different features compared to selection of Ardam et al.
( Ardam et al., 2021 ). Their research estimates the two-phase gas
multiplier in two-phase adiabatic air-water ﬂ ow in micro- ﬁ nned
tubes based on Random Forest model.
3.2.3 Modeling of heat exchanger performance
The overall performance of a heat exchanger is typically
measured by the overall heating or cooling heat transfer rate
capacity, which will be dependent on the dimensions of the heat
exchanger, or heat exchanger effectiveness or ef ﬁ ciency, which are
dimension-independent. Various research studies have applied
different machine learning methods to distinct aspects of heat
exchanger performance prediction. Both the work of Li et al. ( Li
et al., 2016 ) and Shojaeefard et al. ( Shojaeefard et al., 2017 ) focused
on the prediction of cooling capacity in heat exchangers. While Li
et al. employed a Response Surface Methodology (RSM)-based
Neural
Network
(NN)
model,
Shojaeefard
et
al.
evaluated
different Arti ﬁ cial Neural Network (ANN) structures in their
model. On the other hand, Krishnayatra et al. ( Krishnayatra
et al., 2020 ) and Roy and Majumder ( Roy and Majumder, 2019 )
investigate the prediction of performance parameters in shell and
tube heat exchangers, including exergetic plant ef ﬁ ciency, energetic
cycle ef ﬁ ciency, electric power, fouling factor, and cost, utilizing the
FFBN algorithm with tube con ﬁ gurations, ﬂ uid type, surface area,
and
temperatures
as
input
parameters.
Furthermore,
Muthukrishnan et al. ( Muthukrishnan et al., 2020 ) developed a
Support Vector Machine (SVM) in shell and tube heat exchangers to
predict the heat transfer rate, with results showing the superior
prediction accuracy of SVM over mathematical models. The
consolidated database is from the experiments conducted by
Wang et al. ( Wang et al., 2006 ). The main differences between
these studies lie in the focus of the research (such as cooling capacity,
ef ﬁ ciency, heat transfer rate, etc.), the prediction model used (such
as RSM-based NN, ANN, FFBN, SVM, etc.), and the type of heat
exchanger studied.
Turning our attention to predicting coef ﬁ cient of performance
of heat exchanger systems, it is also clear that this segment has been
at the forefront of integrating innovative machine learning
approaches in research. Bhattacharya et al. ( Bhattacharya et al.,
2022 )
developed
and
validated
a
model
that
combines
Convolutional Neural Networks (CNN) with Gated Recurrent
Units in a State Space Model framework. Their work aimed to
predict the intricate dynamics of heat exchangers observed in vapor
Frontiers in Energy Research
frontiersin.org
13
Zou et al.
10.3389/fenrg.2023.1294531

---

compression cycles in heat exchanger. The model processed inputs
like _ m a in , P atm , T in , RH in , _ m ref in , h in , P out , h out and produced
predictions for p inlet , p outlet , h inlet , h outlet , Q total , _ m total .Their research
demonstrated remarkable accuracy, with the maximum percentage
error being limited to 0.2%. In addition, Chen et al. ( Chen et al.,
2023 ) constructed two-year ﬁ eld tests based on an energy pile heat
pump system, where in situ results were used as sample points, and
the
measured
ambient
temperature
and
humidity,
room
temperature and humidity, and hourly power consumption were
used as input parameters to predicted coef ﬁ cient of performance.
The results showed that the accuracy was higher than that of the
empirical regression models. Moreover, Zhu et al. ( Zhu et al., 2021a )
investigated the boiling and condensation heat transfer of R134a
refrigerant within microchannels under various conditions. Data
collected from these experiments were utilized to train machine
learning-based arti ﬁ cial neural network models for predicting heat
transfer performance. The models effectively forecasted the heat
transfer coef ﬁ cients for both boiling and condensation processes.
Further, Li et al. ( Li et al., 2023 ) employed four machine learning
methods to anticipate the thermal performance of supercritical
methane ﬂ ow in a Printed Circuit Heat Exchanger. The ANN
proved to be highly precise in forecasting the local heat transfer
coef ﬁ cient and unit pressure drop following hyperparameter
optimization.
3.2.4 Conclusion of heat exchangers modeling
Upon the review of the recent studies using machine learning to
predict various performance indicators for different types of heat
exchangers, several key themes and opportunities for enhancement
emerge. Regarding the interaction of various factors within the
models, it is critical to understand that the reliability and
precision
of
machine
learning
predictions
depend
on
a
comprehensive understanding of the interactions between model
parameters. In many of the reviewed studies, parameters such as the
Reynolds number, Weber number, and the Froude number were
utilized, yet the dynamic interactions between these parameters were
not explicitly elucidated. For example, the interplay between
Reynolds number and Froude number could potentially in ﬂ uence
the
prediction
of
pumping
power
signi ﬁ cantly.
A
deeper
investigation into these correlations could lead to more re ﬁ ned
and precise predictions and ultimately, more effective heat
exchanger
designs.
Employing
methods
such
as
feature
importance analysis or sensitivity analysis could provide more
tangible insights into these interactions.
When
scrutinizing
the
model ’ s
training
and
validation
procedures, it becomes imperative to thoroughly outline each
stage of the process. Regrettably, the comprehensive explanation
of this process, encompassing critical aspects such as the selection of
training and validation datasets, hyperparameter tuning, and
over ﬁ tting
prevention,
is
commonly
absent
in
the
studies
reviewed. This lack of essential information hampers both
reproducibility and potential model enhancement. Therefore,
advancing in this ﬁ eld is reliant on a more transparent and
detailed presentation of these steps.
On this basis, the role of data transparency and reproducibility
cannot be overstated in ensuring the credibility and utility of these
models. Some studies, however, fall short by failing to explicitly state
their data sources or by not providing clear de ﬁ nitions of model
parameters. These omissions could obstruct other researchers ’
understanding and reproduction of the models. Hence, by
improving data openness and providing a more transparent
presentation of model parameters, the ﬁ eld could experience
signi ﬁ cant
advancements,
facilitating
replication
and
model
improvement.
Lastly, when we turn our attention to the exploration of
emerging
techniques,
it
is
clear
that
traditional
machine
learning methods such as Arti ﬁ cial Neural Networks (ANN),
Gradient Boosting Machines (GBM), and Ridge Regression
have been well documented. However, a noticeable gap exists
in the exploration and application of more recent machine
learning methodologies. Techniques like deep learning and
reinforcement learning, which have shown promise in various
other disciplines, could potentially enhance predictive capabilities
and robustness in heat exchanger performance prediction. This
untapped potential area is, thus, deserving of further, in-depth
investigation.
3.3 Fouling factor
The fouling factor is an index that measures the unit thermal
resistance of solid sediments deposited on heat exchange surfaces
and reduces the overall heat transfer coef ﬁ cient of the heat
exchanger ( Müller-Steinhagen, 1999 ). Fouling deposits that clog
the channels of compact heat exchangers will increase pressure
drops and reduce ﬂ ow rates, resulting into poor heat transfer and
ﬂ uid ﬂ ow performance ( Asadi et al., 2013 ). Table 7 lists the details of
the literature dealing with fouling factors of heat exchangers. A
summary of these investigations is discussed in this subsection. For
predicting the fouling factor, Hosseini et al. ( Hosseini et al., 2022 )
estimated the fouling factor through four machine learning
methods: Gaussian Process Regression (GPR), Decision Trees
(DT), Bagged Trees, and Support Vector Regression (SVR). The
database was collected from experiments, and the model inputs were
the operation time, the surface temperature, the ﬂ uid velocity, the
ﬂ uid density, the ﬂ uid temperature, and the equivalent diameter,
selected
based
on
Pearson ’ s
correlation
analysis.
Mohanty
( Mohanty, 2017 ) estimated the temperature difference on the
tube and shell sides of a shell-and-tube heat exchanger, as well as
the heat exchanger ef ﬁ ciency as the outputs of a fouling factor-based
ANN with network structure 6-5-4-2.
For estimating the fouling factor, Kuzucanl ı et al. ( Kuzucanl ı
et al., 2022 ) predicted the behavior of the overall heat transfer
coef ﬁ cient and of the in plate heat exchangers. It is noteworthy that
this work introduced the classi ﬁ cation solution. The dataset was
collected from the experiment with variable ﬂ ow rates and inlet
temperatures as input parameters. In a similar work, Sundar et al.
( Sundar et al., 2020 ) predicted the fouling factor based on deep
learning. A total of 15,600 samples were collected in a database,
using the inlet ﬂ uid temperatures, the ratio of fouled ﬂ uid ﬂ ow rates
to ﬂ ow rates under clean circumstances, and the outlet temperatures
(gas and ﬂ uid) as inputs.
Machine learning methods have proven effective in modeling
and predicting the fouling factor in heat exchangers, a measure that
signi ﬁ cantly impacts thermal performance. Techniques such as
Gaussian Process Regression, Decision Trees, Bagged Trees, and
Frontiers in Energy Research
frontiersin.org
14
Zou et al.
10.3389/fenrg.2023.1294531

---

Support Vector Regression have been used, leveraging operational
parameters like operation time, surface temperature, ﬂ uid velocity,
and more. These methods have shown acceptable prediction
accuracy, demonstrating machine learning ’ s effectiveness in this
ﬁ eld. Additionally, machine learning has been effective in predicting
fouling ’ s impact on other parameters, like temperature difference
and heat exchanger ef ﬁ ciency. While existing algorithms have been
primarily used, there ’ s potential for new machine learning
algorithms to further improve fouling factor prediction.
3.4 Refrigerant thermodynamic properties
The conventional prediction of the refrigerant thermodynamic
properties is usually carried out by means of empirical, theoretical,
and numerical models. Although these methods have been
successfully applied in many cases, their numerical modeling still
suffers from computational issues in dealing with the complex
molecular structure of refrigerants ( Meghdadi Isfahani et al.,
2017 ; Alizadeh et al., 2021a ). Table 8 lists several machine
learning prediction models of the thermodynamic properties of
refrigerants available in the literature, which are brie ﬂ y described
in this subsection.
In literature, neural network models have been widely employed
by many researchers for the prediction of refrigerant properties. For
example, Gao et al. ( Gao et al., 2019 ), Wang et al. ( Wang et al., 2020 ),
Zolfaghari and Youse ﬁ ( Zolfaghari and Youse ﬁ , 2017 ), Nabipour
( Nabipour, 2018 ) employed ANN to predict the thermodynamic
properties, such as, P red , 1 − T r , ω , Pc / Pcr , etc. They employed
different parameters to investigate the prediction performance, for
instance, Wang et al. ( Wang et al., 2020 ) introduced ANNs to
estimate the viscosity and the thermal conductivity, using the
reduced pressure ( P red ), the reduced temperature ( T r ), the molar
mass ( M ), and the acentric factor ( ω ) as inputs. Similarly, Zolfaghari
and Youse ﬁ ( Zolfaghari and Youse ﬁ , 2017 ) developed an ANN to
predict the density of sixteen lubricant/refrigerant mixtures,
considering a total of 3,961 data points from the literature. In
this study, the temperature ( T ), the pressure ( P ), the molar
fraction ( x ), the total molecular weight ( M w ), and the average
boiling temperature ( T b ) of pure refrigerants were considered as
input parameters.
Shifting away from the singular prediction model approach,
numerous studies have adopted a more extensive analysis by
examining multiple prediction models. Several studies have
embraced a more comprehensive analysis by investigating more
than one prediction model; for example, Zhi et al. ( Zhi et al., 2018 )
developed three prediction models of viscosity based on ANFIS,
RBFNN,
and
BPNN
for
six
pure
refrigerants,
speci ﬁ cally
R1234ze(E), R1234yf, R32, R152a, R161, and R245fa in the
saturated liquid state. It is reported that a total of 1,089 data
points were collected from the literature, of which 80% were
allocated to training and 20% to testing, while the algorithm
inputs were temperature, pressure, and liquid density. Results
demonstrate
that
the
ANFIS
algorithm
shows
the
highest
prediction accuracy.
Upon reviewing the impressive statistics presented in Table 8 , it
is evident that machine learning has proven to be an invaluable tool
for predicting the thermodynamic properties of refrigerants. A
common thread across the studies indicates that factors such as
temperature, pressure, and density often serve as inputs for these
predictive models. However, we observe variations in the algorithms
used and the speci ﬁ c properties predicted. This could be attributed
to the unique characteristics of the refrigerants studied and the
speci ﬁ c objectives of each study. While these models demonstrate
impressive prediction accuracy, it is crucial to acknowledge that
model performance varies depending on the refrigerant and
property in question. A broader observation reveals a notable
trend toward using machine learning in refrigerant property
prediction, which presents opportunities for further exploration.
Future work could include comprehensive comparative studies of
these different machine learning algorithms, considering their
strengths and weaknesses in various scenarios. There is also
potential for integrating these machine learning models with
other
computational
tools
for
more
robust
and
accurate
predictions. Furthermore, as the ﬁ eld continues to evolve, there
may be scope to explore new machine-learning techniques and
develop novel approaches for predicting the thermodynamic
properties of refrigerants.
3.5 Flow patterns
Two-phase ﬂ ow is critical in many chemical processes, heat
transfer, and energy conversion technologies. The ﬂ ow pattern in
two-phase ﬂ ow has a critical role in heat transfer coef ﬁ cient and
pressure drop, because the physics governing the pressure drop
and the heat transfer is intrinsically linked to the local distribution
of the liquid and vapor phases ( Cheng et al., 2008 ). Recently, the
prediction of ﬂ ow patterns based on machine learning has received
growing attention. Table 9 summarizes the studies about ﬂ ow
pattern recognition based on machine learning reported in the
present work. Identifying ﬂ ow patterns is crucial in ﬂ uid
mechanics, employing various methods. High-speed cameras
offer direct visual insight but are limited to transparent media.
Gamma rays can analyze opaque ﬂ uids but raise safety concerns.
Pressure sensors can infer ﬂ ow patterns from pressure changes,
albeit with interpretational challenges. The Continuous Wave
Doppler technique measures particle velocities using frequency
shifts but requires particles or bubbles for measurement. The
appropriate method hinges on factors like ﬂ ow type, ﬂ uid
transparency, piping material, safety, and the depth of analysis
required.
Some studies identi ﬁ ed the ﬂ ow regimes using the high-speed
cameras, Shen et al. ( Shen et al., 2020 ) Ahmad et al. ( Ahmad et al.,
2022 ), Giri Nandagopal et al. ( Giri Nandagopal and Selvaraju, 2016 )
and Giri Nandagopal et al. ( Nandagopal et al., 2017 ) investigate the
ﬂ ow pattern recognition through high-speed cameras. For instance,
Giri Nandagopal et al. ( Nandagopal et al., 2017 ) investigated the
same liquid-liquid system in a circular microchannels of 600 μ m
diameter as the con ﬂ uence angle of the two ﬂ uids was varied in the
range 10 – 170 degrees, in order to predict the ﬂ ow pattern maps
using the con ﬂ uence angle and the super ﬁ cial velocities of the two
liquids as input. The algorithms considered could identify slug ﬂ ow,
bubble ﬂ ow, deformed ﬂ ow, elongated slug ﬂ ow, deformed ﬂ ow, and
strati ﬁ ed ﬂ ow. The results showed that GRNN gives the best
prediction accuracy again.
Frontiers in Energy Research
frontiersin.org
15
Zou et al.
10.3389/fenrg.2023.1294531

---

Instead of using a high-speed camera to record the ﬂ ow regimes
included in the datasets, some studies used gamma rays to construct
the database. For example, Roshani et al. ( Roshani et al., 2017 )
identi ﬁ ed the ﬂ ow regimes by means of the multi-beam gamma ray
attenuation technique. In this study, the outputs of two detectors
are introduced as input parameters into the RBF models in order to
predict the ﬂ ow regimes. Similarly, Hanus et al. ( Hanus et al., 2018 )
used the gamma-ray attenuation technology to identify ﬂ ow
regimes
and
generate
input
data
for
the
algorithm.
In
particular, nine features obtained from the signal analysis were
selected as inputs and applied to six different machine-learning
methods. The results showed a promising accuracy for all the
methods considered.
In contrast to those described above, some studies employed
other methods, such as pressure sensors, ultrasound, and a new
concept (take-off ratio). For example, Godfrey Nnabuife et al.
( Godfrey et al., 2021 ) used Deep Neural Networks (DNNs)
operating
on
features
extracted
from
Continuous
Wave
Doppler Ultrasound (CWDU) to recognize the ﬂ ow regimes of
an unknown gas-liquid ﬂ ow in an S-shaped riser. A Twin-
window Feature Extraction algorithm generates the vectors
that contain all the information used as input of the Deep
NN, reducing the amount of input data and eliminating the
noise. The identi ﬁ ed ﬂ ow regimes are annular, churn, slug, and
bubbly ﬂ ow. The results show the highest prediction accuracy,
which is better in comparison with that of four conventional
machine learning methods: AdaBoost, Bagging, Extra Trees, and
DT. Khan et al. ( Khan et al., 2022 ) developed CNN to identify the
ﬂ ow regimes in air-water ﬂ ow in a horizontal pipe with a 5 cm
inner diameter, using the scalograms obtained from pressure
detectors as input database. Differently from the described above,
Giannetti et al. ( Giannetti et al., 2020 ) introduced the concept of
take-off ratio to develop an ANN to predict the two-phase ﬂ ow
distribution in microchannel heat exchangers based on a limited
amount of input information. The concept of take-off ratio is
based on Prigogine ’ s theorem of minimum entropy generation
( Onsager, 1931 ; Prigogine and Van Rysselberghe, 1963 ). As a
result, the 4-3-3-3-1 architecture achieves the highest prediction
accuracy reported.
Machine learning has increasingly been applied to predict
and understand ﬂ ow patterns in two-phase ﬂ ow systems, a topic
of substantial signi ﬁ cance across various ﬁ elds, from chemical
processes to energy conversion technologies. The range and
diversity of research in this domain underline the complex
interplay
between
the
physical
parameters
governing
the
pressure drop and heat transfer, which are intricately related
to the local distribution of liquid and vapor phases. Key to this
research is the use of machine learning to identify and distinguish
different ﬂ ow patterns accurately. This has been addressed using
diverse techniques, such as CNNs, DL, and various types of
ANNs,
including
the
PNN,
GRNN,
and
ANFIS.
These
methods
have
demonstrated
high
degrees
of
prediction
accuracy in their respective applications, offering promising
advancements in the ﬁ eld. The generation of input data for
these machine learning models has employed an array of
innovative methodologies, such as high-speed camera image
capturing
and
the
use
of
the
multi-beam
gamma
ray
attenuation technique. Some studies have further expanded
upon this by introducing novel concepts, such as the take-off
ratio, which applies Prigogine ’ s theorem of minimum entropy
generation to predict two-phase ﬂ ow distribution. Other research
has veered towards the use of Deep Neural Networks (DNNs) to
identify ﬂ ow regimes based on Continuous Wave Doppler
Ultrasound (CWDU) information, exhibiting high prediction
accuracy rates. This move toward the use of DNNs and
similar methods demonstrates the ﬁ eld ’ s continuous evolution
and the trend toward more sophisticated, precise prediction
models.
3.6 Structured approach to model selection
in machine learning: A guide
The selection and evaluation of machine learning algorithms
necessitates a comprehensive and multi-faceted approach, involving
numerous interdependent steps and considerations. This section
delineates a systematic methodology devised to aid practitioners in
judiciously selecting the pertinent machine learning algorithm
tailored for a speci ﬁ c problem domain.
1. Problem
De ﬁ nition:
The
preliminary
step
involves
a
comprehensive understanding of the problem landscape. This
encompasses identifying the nature of the problem — be it a
classi ﬁ cation, regression, clustering, or another variant.
2. Exploratory Data Analysis: Exploratory Data Analysis is the
initial phase of understanding data, aiming to summarize its
main characteristics, often visually. This phase includes assessing
feature distributions through histograms or boxplots to spot
skewness,
understanding
data
sparsity
with
matrix
visualizations,
detecting
outliers
via
scatter
plots
or
Interquartile Range methods, and discerning missing value
patterns with heatmaps or bar charts. Correlation matrices
and pair plots can reveal relationships between variables.
Dimensionality
reduction
techniques,
such
as
Principal
Component Analysis or t-distributed Stochastic Neighbor
Embedding, provide a compressed visual perspective on multi-
dimensional data.
3. Data Pre-processing: Based on Exploratory Data Analysis
ﬁ ndings, data pre-processing re ﬁ nes the dataset for modeling.
Feature engineering may involve creating polynomial features,
encoding categorical variables, or extracting time-based metrics.
Outliers could be capped, transformed, or removed entirely.
Standard practices also include scaling features using methods
like Minimum-Maximum or z-score normalization. Categorical
data often require encoding techniques such as one-hot or
ordinal. Finally, data may be split into training, validation,
and test sets to evaluate the model ’ s performance effectively.
4. Evaluation Metric Selection: The choice of an evaluation metric
should align closely with both the problem de ﬁ nition and
organizational
objectives.
For
instance,
in
classi ﬁ cation
problems, metrics like accuracy, MAE, MRE, etc. may be
considered.
5. Comparative Model Assessment: Employing techniques like
cross-validation,
the
performance
of
multiple
candidate
algorithms should be rigorously compared to ascertain the
most effective model based on the validation dataset.
Frontiers in Energy Research
frontiersin.org
16
Zou et al.
10.3389/fenrg.2023.1294531

---

6. Hyperparameter Optimization: Subsequent to model selection,
hyperparameter tuning is conducted to further re ﬁ ne the
performance of the selected models.
7. Validation
and
Testing:
Final
performance
evaluation
is
conducted using an independent test set to ascertain the
generalizability of the model and to mitigate the risk of
over ﬁ tting.
### 4 Limitations and potential solutions
Despite the remarkable potential and superior performance of
machine learning techniques compared to traditional computational
methods, their unique features, such as a tendency towards
over ﬁ tting and interpretability can present hurdles in their
application
within
heat
exchanger
systems.
The
ensuing
discussion will delve into the primary issues in deploying
machine learning strategies in the process of modeling heat
exchangers, alongside exploring possible solutions.
4.1 Over ﬁ tting
Like most probabilistic models, the issues of over ﬁ tting and
under- ﬁ tting
are
unavoidable
in
machine
learning
models
( Dobbelaere et al., 2021 ). Over ﬁ tting refers to the prediction
accuracy being extremely high in the training dataset, while the
performance on the testing dataset is unsatisfactory ( Dietterich,
1995 ).
There
are
multiple
potential
explanations
of
the
phenomenon, such as noise over-learning on the training set
( Paris et al., 2003 ), hypothesis complexity ( Paris et al., 2003 ), and
multiple comparison procedures ( Jensen and Cohen, 2000 ).
In order to mitigate over ﬁ tting problems, it is recommended
to introduce the following strategies: a) Early stopping ( Jabbar
and Khan, 2015 ), which requires de ﬁ ning the criteria of stopping
functions, for instance, monitoring the performance of the model
on a validation set during the training process. The training is
stopped when the error on the validation set starts to increase,
which is a sign of over ﬁ tting. The validation set is a small portion
of the training data set aside to check the model ’ s performance
during training. b) Network structure optimization ( Dietterich,
1995 ), which involves tuning the architecture of the neural
network to ﬁ nd the most ef ﬁ cient structure. For example, one
could experiment with different numbers of layers or different
numbers of neurons per layer. Additionally, pruning methods
can be used to reduce the complexity of decision trees or neural
networks by eliminating unnecessary nodes. c) Regularization
( Jabbar and Khan, 2015 ), similar to penalty methods, is used to
reduce the in ﬂ uence of noise. This term discourages the model
from assigning too much importance to any one feature, reducing
the risk of over ﬁ tting. In conclusion, while several studies in
Tables 1 – 8 have incorporated the early stopping and network
structure
optimization
techniques,
it
is
unclear
if
they
signi ﬁ cantly reduced over ﬁ tting. Further evaluation of these
methods ’ effectiveness in the studies mentioned might offer
more insights. Regularization, however, seems to be less
frequently employed, based on our review.
4.2 Interpretability
Machine learning methods are essentially black box models,
where data analysis can be understood as a pattern recognition
process ( Dobbelaere et al., 2021 ). According to Vellido ( Vellido
et al., 2012 ), interpretability refers to the ability to assess and explain
the reasoning behind machine learning model decisions, which is
one of the most signi ﬁ cant qualities machine learning methods
should achieve in practice. Model hyperparameters, such as node
optimization in arti ﬁ cial neural networks, are key elements in
constructing an effective model. The selection and tuning of
these hyperparameters typically have a signi ﬁ cant impact on the
performance of the model. However, for these types of models, the
analysis usually focuses on prediction accuracy rather than the
interpretability of the model ( Feurer and Hutter, 2019 ). To
implement
interpretability,
dimensionality
reduction
can
be
introduced for supervised and unsupervised ( Azencott, 2018 )
problems through feature selection and feature extraction ( Dy
et al., 2000 ; Guyon and Elisseeff, 2003 ; Guyon et al., 2008 ). In
addition, Vellido ( Alcacena et al., 2011 ) stated that information
visualization is a feasible solution to interpret the machine learning
models such as Partial Dependency Plots (PDP) ( Greenwell, 2017 )
and Shapley Additive explanation (SHAP) ( Mangalathu et al., 2020 ).
It is important to build models that can self-learn to recognize
patterns and self-evaluate.
In the latest study, Xie et al. ( Xie et al., 2022 ) introduced a
mechanistic data-driven approach called dimensionless learning.
It identi ﬁ es key dimensionless ﬁ gures and governing principles
from limited data sets. This physics-based method simpli ﬁ es high-
dimensional spaces into forms with a few interpretable parameters,
streamlining complex system design and optimization. It also
states that the processes could ﬁ nd very useful application in
heat exchanger modeling and heat exchanger experimental data
characterization. This method unveils scienti ﬁ c knowledge from
data through two processes. The ﬁ rst process embeds the principle
of dimensionless invariance (i.e., physical laws being independent
of the fundamental units of measurement) into a two-tier machine
learning framework. It discovers the dominating dimensionless
numbers and scaling laws from noisy experimental data of
complex physical systems. The subjects of investigation include
Rayleigh – Bénard convection, vapor-compression dynamics in the
process of laser melting metals, and pore formation in 3D printing.
The second process combines dimensionless learning with
a
sparsity-promoting
technique
to
identify
dimensionless
homogeneous differential equations and dimensionless numbers
from data. This method can enhance the physical interpretability
of machine learning models.
4.3 Data quality and quantity
The prediction of parameters based on machine learning can
provide a reference for scienti ﬁ c research and practical applications
to both researchers and engineers However, it is worth mentioning
that dealing with a database containing too many outsider data
points can generate system errors. Compared with an extensive
database, machine learning is more sensitive to a small database,
Frontiers in Energy Research
frontiersin.org
17
Zou et al.
10.3389/fenrg.2023.1294531

---

which can in ﬂ uence machine learning models ( Pourkiaei et al.,
2016 ).
It is possible to increase the number of data points ( Dietterich,
1995 ), delete the outsider data points, and use algorithms for
anomaly detection, such as the principal component analysis
(PCA) algorithm ( Thombre et al., 2020 ) and LSTM ( Zhang
et al., 2019 ). In addition, it is also possible to carefully examine
the data for stable, reliable, and repeatable data ( Zhou et al., 2020 ).
Although decades of modeling, simulations, and experiments have
produced several datasets about heat exchangers, they are often
archived in research laboratories or companies and are not open
access.
Lindqvist et al. ( Lindqvist et al., 2018 ) introduced the
employment of structured and adaptive sampling methodologies.
Structured sampling techniques, such as Latin Hypercube Sampling,
systematically distribute sample points throughout the design space,
thereby providing a robust approach to experimental design.
Conversely, adaptive sampling dynamically modi ﬁ es the location
of sample points contingent on the predictive outcomes of the
model, thereby optimizing model performance.
4.4 Model generalization
Model generalization refers to the ability of a machine
learning model to adapt properly to new, unseen data drawn
from the same distribution as the one used to train the model
( Bishop and Nasrabadi, 2006 ). It is a critical aspect of machine
learning models, particularly in complex ﬁ elds such as ﬂ uid
dynamics
and
heat
transfer,
where
phenomena
can
be
in ﬂ uenced by a multitude of factors. A model ’ s generalization
capability determines its utility and applicability in real-world
scenarios beyond the con ﬁ nes of the training data. However,
achieving good generalization is a signi ﬁ cant challenge and often
requires careful model design and validation strategies. When
applying machine learning methods outside the scope of the
database, outputs will be unreasonable. A limited training dataset
determines the scope of the application.
When assessing unknown data points via a predictive model,
users must ensure that these data points lie within the model ’ s
operational domain. “ Unknown data points ” typically represent data
not previously encountered during the model ’ s training process. As
they are excluded from the training dataset, the model extrapolates
its learned patterns to generate predictions for these data points.
These unknown data points are instrumental in evaluating the
model ’ s generalization capabilities. However, should these data
points fall outside the model ’ s operational domain, the reliability
of the resultant predictions could be undermined. To maintain the
trustworthiness of computations under such circumstances, it is
recommended to either augment the training database to encompass
a broader data spectrum or cross-validate the predicted values
employing alternative credible methodologies ( Azencott, 2018 ).
### 5 Emerging applications
Here, the emerging heat exchanger applications involving
machine learning will be discussed, including the novel nano ﬂ uid
mixture
modeling,
heat
exchanger
design,
and
topology
optimization.
5.1 Nano ﬂ uid
Nano ﬂ uids are widely used in solar collectors, heat exchangers,
heat pipes, and other energy systems ( Ramezanizadeh et al., 2019 ).
The presence of nanoparticles within the ﬂ uid can enhance the
thermophysical properties of the ﬂ uid to bene ﬁ t the heat transfer
behavior within the system. Currently, several machine learning
models have been introduced to predict the thermodynamic
properties of hybrid nano ﬂ uids ( Maleki et al., 2021 ). According
to the Web of Science database, about 3% of nano ﬂ uid research
papers published in 2019 involved machine learning, with an
increasing trend (T. Ma et al., 2021 ).
In the literature, several machine-learning models have been
applied to heat exchangers containing nano ﬂ uids ( Naphon et al.,
2019 ; Ahmadi et al., 2020 ; Gholizadeh et al., 2020 ; Hojjat, 2020 ;
Kumar and Rajappa, 2020 ; Alimoradi et al., 2022 ). Nano ﬂ uids
involve
complex
physical,
chemical,
and
ﬂ uid
dynamic
phenomena, and traditional modeling and analysis methods may
face challenges. However, machine learning, as a data-driven
approach, can help address the complex problems in nano ﬂ uid
research by learning and discovering patterns and correlations in the
data ( Ma et al., 2021 ). For instance, Cao et al. ( Cao et al., 2022 )
employed machine learning to simulate the electrical performance of
photovoltaic/thermal
(PV/T)
systems
cooled
by
water-based
nano ﬂ uids. Alizadeh et al. ( Alizadeh et al., 2021a ) proposed a
novel
machine
learning
approach
for
predicting
transport
behaviors in multiphysics systems, including heat transfer in a
hybrid nano ﬂ uid ﬂ ow in porous media. Another study by
Alizadeh et al. ( Alizadeh et al., 2021b ) used an arti ﬁ cial neural
network for predictive analysis of heat convection and entropy
generation in a hybrid nano ﬂ uid ﬂ owing around a cylinder
embedded in porous media. Machine learning also can assist in
analyzing large amounts of experimental data to extract useful
information
and
trends,
accelerating
research
progress.
For
example, machine learning algorithms can be used to predict and
optimize the surface properties, dispersibility, and ﬂ ow behavior of
nanoparticles ( El-Amin et al., 2023 ). Moreover, machine learning
can be used for simulating and optimizing the design and
performance of the system containing nano ﬂ uid providing more
ef ﬁ cient solutions (T. Ma et al., 2021 ).
At the nanoscale, the conventional principles of ﬂ uid mechanics
and heat transfer may not hold true, thus necessitating innovative
theories to decode the behavior of nano ﬂ uids. While machine
learning could reveal unseen patterns and correlations within
data, it does not guarantee the applicability of these trends under
nanoscale constraints. Nano ﬂ uidic research, given its complex
nature, requires experimental veri ﬁ cation for the predictions
formulated
by
machine
learning
models.
However,
this
veri ﬁ cation process often demands sophisticated instrumentation,
advanced methodologies, and considerable ﬁ nancial resources,
which may pose signi ﬁ cant challenges and potentially exceed the
capabilities of numerous research groups. Nano ﬂ uid systems are
marked by a high degree of complexity due to the interaction among
various components such as ﬂ uids, nanoparticles, and interfaces,
Frontiers in Energy Research
frontiersin.org
18
Zou et al.
10.3389/fenrg.2023.1294531

---

thereby rendering the prediction process through machine learning
models extremely challenging. Moreover, nano ﬂ uid research is data-
intensive, and procuring the requisite amount of data can often be
problematic.
5.2 Heat exchangers design and
optimization
Machine learning algorithms can analyze large amounts of data,
identify patterns, and make predictions or decisions without being
explicitly programmed to perform the task. This ability to learn from
data makes machine learning particularly useful in optimization
problems, where the goal is to ﬁ nd the best solution among a set of
possible solutions. It indicates that it can be a powerful tool for
dealing with various engineering issues. It is reported that machine
learning can potentially optimize the topology structure of heat
exchangers. According to Fawaz ( Fawaz et al., 2022 ), machine
learning algorithms can be combined with a density-based
topology algorithm, which is mainly aimed at structural design at
the present stage ( Sosnovik and Oseledets, 2019 ; Abueidda et al.,
2020 ; Chandrasekhar and Suresh, 2021 ; Chi et al., 2021 ). Moreover,
few studies are coupled with ML and Topology (TO) for HXs, which
may be related to the complexity of coupled heat transfer
(particularly the ﬂ uid ﬂ ow part) and the complexity of HXs
structure
( Fawaz
et
al.,
2022 ).
Michalski
( Michalski
and
Kaufman, 2006 ) introduced the Learnable Evolution Model
(LEM), containing the hypothesis generation and instantiation to
create new designs based on machine learning methods, which can
automatically search for the highest capacity heat exchangers under
given technical and environmental constraints. LEM has a wide
range of potential applications, especially in complex domains,
optimization, or search problems ( Michalski, 2000 ). The results
of the methods have been highly promising, producing solutions
exceeding the performance of the best human designs ( Michalski
and Kaufman, 2006 ).
Although machine learning holds signi ﬁ cant promise for the
design and optimization of heat exchangers, however, it is crucial to
acknowledge that the application of these techniques in this ﬁ eld is
still in its infancy. The intricate physical phenomena and
interactions
involved
in
heat
exchanger
systems
present
a
signi ﬁ cant challenge for machine learning models. Despite the
potential, there are substantial hurdles to overcome. Future work
in this ﬁ eld should concentrate on enhancing the interpretability of
machine learning models, as previously mentioned. Additionally,
efforts should be made to develop methods for generating novel
design concepts and to create high-quality datasets for training these
models. By addressing these challenges, we can better harness the
power of machine learning in the design and optimization of heat
exchangers.
### 6 Conclusion
This paper provides a comprehensive review of heat exchanger
modeling based on machine learning methods, drawing on literature
published over the past 8 years. The review evidences a clear
expansion of this ﬁ eld, with a signi ﬁ cant publication growth rate
observed after 2018. As shown in Figure 4 , neural networks have
been widely implemented, accounting for about 56% of the
literature. This is attributed to their high prediction accuracy and
powerful parallel and distributed processing capabilities. The paper
systematically explores the entire gamut of heat exchanger modeling
based on machine learning methods, focusing on types of
algorithms, input parameters, output parameters, and error
analysis.
These
insights
can
guide
researchers
in
selecting
appropriate
machine
learning
models
for
various
heat
exchangers,
predicting
fouling
factors,
and
thermodynamic
properties of refrigerants, tailored to their speci ﬁ c objectives.
Despite the promising performance of machine learning
methods under the right database conditions, several limitations
exist, including data over ﬁ tting, anomaly processing, limited scope,
FIGURE 4
The proportion of publications on various machine learning methods.
Frontiers in Energy Research
frontiersin.org
19
Zou et al.
10.3389/fenrg.2023.1294531

---

and low interpretability. Accordingly, feasible schemes have been
introduced to mitigate these limitations. The paper also emphasizes
the potential of Dimensionless Learning as discussed in Section 3 .
Speci ﬁ cally, incorporating the interplay between dimensionless
numbers such as Re, We, and Fr numbers could provide a more
generalizable and physically intuitive understanding of heat
exchanger performance and ﬂ uid ﬂ ow behavior. Furthermore, an
area that is conspicuously underrepresented in the current literature
is the modeling of surface roughness using machine learning
methods, presenting a clear opportunity for future research.
Finally,
two emerging
areas,
nano ﬂ uids
in
new
energy
applications and heat exchanger design optimization, are also
discussed. The data-driven approach to machine learning offers
new possibilities for thermal analysis of ﬂ uids, cycles, and heat
exchangers with faster calculation and higher prediction accuracy.
The information provided in this paper will greatly bene ﬁ t
researchers who aim to utilize machine learning methods in the
ﬁ eld of heat exchangers and thermo- ﬂ uid systems in general.
### Author contributions
JZ:
Writing – original
draft,
Writing – review
and
editing,
Investigation, Project administration. TH: Writing – review and
editing,
Methodology.
JA:
Writing – review
and
editing,
Visualization.
LH:
Writing – review
and
editing,
Conceptualization, Funding acquisition, Methodology, Project
administration, Supervision. JC: Writing – review and editing,
Conceptualization, Methodology, Supervision.
### Funding
The author(s) declare that ﬁ nancial support was received for the
research, authorship, and/or publication of this article. This work
was supported by the Natural Science Foundation of the Higher
Education Institutions of Jiangsu Province, China (Grant No.
21KJB470011),
State
Key
Laboratory
of
Air-conditioning
Equipment
and
System
Energy
Conservation
Open
Project
(Project No. ACSKL2021KT01) and the Research Development
Fund (RDF 20-01-16) of Xi ’ an Jiaotong-Liverpool University. For
the purpose of open access, the authors have applied a Creative
Commons Attribution (CC-BY) licence to any Author Accepted
Manuscript version arising from this submission.
### Con ﬂ ict of interest
The authors declare that the research was conducted in the
absence of any commercial or ﬁ nancial relationships that could be
construed as a potential con ﬂ ict of interest.
### Publisher ’ s note
All claims expressed in this article are solely those of the authors and
do not necessarily represent those of their af ﬁ liated organizations, or
those of the publisher, the editors and the reviewers. Any product that
may be evaluated in this article, or claim that may be made by its
manufacturer, is not guaranteed or endorsed by the publisher.
### References
Abueidda, D. W., Koric, S., and Sobh, N. A. (2020). Topology optimization of 2D
structures with nonlinearities using deep learning. Comput. Struct. 237, 106283. doi:10.
1016/j.compstruc.2020.106283
Abu-Khader, (2012). Plate heat exchangers: recent advances. Renew. Sustain. Energy
Rev. 16, 1883 – 1891. doi:10.1016/j.rser.2012.01.009
Adelaja, A. O., Dirker, J., and Meyer, J. P. (2017). Experimental study of the pressure drop
during condensation in an inclined smooth tube at different saturation temperatures. Int.
J. Heat Mass Transf. 105, 237 – 251. doi:10.1016/j.ijheatmasstransfer.2016.09.098
Ahmad, H., Kim, S. K., Park, J. H., and Sung, Y. J. (2022). Development of two-phase
ﬂ ow regime map for thermally stimulated ﬂ ows using deep learning and image
segmentation
technique.
Int.
J.
Multiph.
Flow
146,
103869.
doi:10.1016/j.
ijmultiphase ﬂ ow.2021.103869
Ahmadi, M. H., Kumar, R., Mamdouh El Haj Assad, and Phuong Thao Thi Ngo,
(2021). Applications of machine learning methods in modeling various types of heat
pipes: a review. J. Therm. Analysis Calorim. 146, 2333 – 2341. Springer Science and
Business Media B.V. doi:10.1007/s10973-021-10603-x
Ahmadi, M. H., Mohseni-Gharyehsafa, B., Ghazvini, M., Goodarzi, M., Jilte, R. D.,
and Kumar, R. (2020). Comparing various machine learning approaches in modeling
the dynamic viscosity of CuO/water nano ﬂ uid. J. Therm. Analysis Calorim. 139 (4),
2585 – 2599. doi:10.1007/s10973-019-08762-z
Alcacena, V., Alfredo, J. D. M., Rossi, F., and Lisboa, P. J. G. (2011). “ Seeing is believing: the
importance of visualization in real-world machine learning applications, ” in Proceedings:
19th European Symposium on Arti ﬁ cial Neural Networks, Computational Intelligence and
Machine Learning, ESANN 2011, Bruges, Belgium, April 27-28-29, 2011, 219 – 226.
Alimoradi, H., Eskandari, E., Pourbagian, M., and Shams, M. (2022). A parametric
study of subcooled ﬂ ow boiling of Al2O3/water nano ﬂ uid using numerical simulation
and arti ﬁ cial neural networks. Nanoscale Microscale Thermophys. Eng. 26 (2 – 3),
129 – 159. doi:10.1080/15567265.2022.2108949
Alizadeh, R., Abad, J. M. N., Ameri, A., Mohebbi, M. R., Mehdizadeh, A., Zhao, D.,
et al. (2021a). A machine learning approach to the prediction of transport and
thermodynamic processes in multiphysics systems - heat transfer in a hybrid
nano ﬂ uid ﬂ ow in porous media. J. Taiwan Inst. Chem. Eng. 124, 290 – 306. doi:10.
1016/j.jtice.2021.03.043
Alizadeh, R., Javad Mohebbi Najm Abad, Fattahi, A., Mohebbi, M. R., Hossein
Doranehgard, M., Larry, K. B. Li, et al. (2021b). A machine learning
approach to predicting the heat convection and thermodynamics of an
external ﬂ ow of hybrid nano ﬂ uid. J. Energy Resour. Technol. 143 (7). doi:10.
1115/1.4049454
Amal ﬁ , R. L., and Kim., J. (2021). “ Machine learning-based prediction methods for
ﬂ ow boiling in Plate Heat exchangers, ” in InterSociety Conference on Thermal and
Thermomechanical Phenomena in Electronic Systems, ITHERM, San Diego, California,
June-2021 (IEEE Computer Society), 1131 – 1139. doi:10.1109/ITherm51669.2021.
9503302
Amal ﬁ , R. L., Vakili-Farahani, F., and Thome, J. R. (2016a). Flow boiling and
frictional
pressure
gradients
in
Plate
Heat
exchangers.
Part
1:
review
and
experimental database. Int. J. Refrig. 61, 166 – 184. doi:10.1016/j.ijrefrig.2015.07.010
Amal ﬁ , R. L., Vakili-Farahani, F., and Thome, J. R. (2016b). Flow boiling and
frictional pressure gradients in Plate Heat exchangers. Part 2: comparison of
literature methods to database and new prediction methods. Int. J. Refrig. 61,
185 – 203. doi:10.1016/j.ijrefrig.2015.07.009
Ardam, K., Naja ﬁ , B., Lucchini, A., Rinaldi, F., and Luigi Pietro Maria Colombo
(2021). Machine learning based pressure drop estimation of evaporating R134a ﬂ ow in
micro- ﬁ n tubes: investigation of the optimal dimensionless feature set. Int. J. Refrig. 131,
20 – 32. November. doi:10.1016/j.ijrefrig.2021.07.018
Ardhapurkar, P. M., and Atrey, M. D. (2015). “ Prediction of two-phase pressure drop
in heat exchanger for mixed refrigerant joule-thomson cryocooler, ” in IOP Conference
Series: Materials Science and Engineering, China, 3-6 August 2015 (IOP Publishing).
101:012111. doi:10.1088/1757-899X/101/1/012111
Asadi, M., Ramin, Dr, and Khoshkhoo, H. (2013). Investigation into fouling factor in
compact heat exchanger. Int. J. Innovation Appl. Stud. 2.
Azencott, C.-A. (2018). Machine learning and genomics: precision medicine versus
patient privacy. Philosophical Trans. R. Soc. A Math. Phys. Eng. Sci. 376 (2128),
20170350. doi:10.1098/rsta.2017.0350
Azizi, S., and Ahmadloo, E. (2016). Prediction of heat transfer coef ﬁ cient during
condensation of R134a in inclined tubes using arti ﬁ cial neural network. Appl. Therm.
Eng. 106, 203 – 210. August. doi:10.1016/j.applthermaleng.2016.05.189
Frontiers in Energy Research
frontiersin.org
20
Zou et al.
10.3389/fenrg.2023.1294531

---

Azizi, S., Ahmadloo, E., and Mohamed, M. A. (2016). Prediction of void fraction for
gas – liquid ﬂ ow in horizontal, upward and downward inclined pipes using arti ﬁ cial
neural network. Int. J. Multiph. Flow 87, 35 – 44. doi:10.1016/j.ijmultiphase ﬂ ow.2016.
08.004
Bhattacharya, C., Chakrabarty, A., Laughman, C., and Qiao, H. (2022). Modeling
nonlinear heat exchanger dynamics with convolutional recurrent networks. IFAC-
PapersOnLine 55 (37), 99 – 106. doi:10.1016/j.ifacol.2022.11.168
Bhutta, M. M. A., Hayat, N., Bashir, M. H., Khan, A. R., Ahmad, K. N., and Khan, S.
(2012). “ CFD applications in various heat exchangers design: a review. ” applied thermal
engineering . Amsterdam, Netherlands: Elsevier Ltd. doi:10.1016/j.applthermaleng.2011.
09.001
Bishop, C. M., and Nasrabadi, N. M. (2006). Pattern recognition and machine
learning. Vol. 4 . Cham: Springer.
Cao, Y., Kamrani, E., Mirzaei, S., Khandakar, A., and Vaferi, B. (2022). Electrical
ef ﬁ ciency of the photovoltaic/thermal collectors cooled by nano ﬂ uids: machine learning
simulation and optimization by evolutionary algorithm. Energy Rep. 8, 24 – 36. doi:10.
1016/j.egyr.2021.11.252
Celebi, M. E., and Aydin, K. (2016). Unsupervised learning algorithms . Cham:
Springer.
Chandrasekhar, A., and Krishnan, S. (2021). TOuNN: topology optimization using
neural networks. Struct. Multidiscip. Optim. 63 (3), 1135 – 1149. doi:10.1007/s00158-
020-02748-4
Chen, J., Gui, W., Dai, J., Jiang, Z., Chen, N., and Xu, Li (2021). A hybrid model
combining mechanism with semi-supervised learning and its application for
temperature prediction in roller hearth kiln. J. Process Control 98, 18 – 29. February.
doi:10.1016/j.jprocont.2020.11.012
Chen, Yu, Kong, G., Xu, X., Hu, S., and Yang, Q. (2023). Machine-learning-based
performance prediction of the energy pile heat pump system. J. Build. Eng. 77, 107442.
doi:10.1016/j.jobe.2023.107442
Cheng, L., Ribatski, G., and Thome, J. R. (2008). Two-phase ﬂ ow patterns and ﬂ ow-
pattern maps: fundamentals and applications. Appl. Mech. Rev. 61 (5). doi:10.1115/1.
2955990
Chi, H., Zhang, Y., Tsz Ling Elaine Tang, Mirabella, L., Dalloro, L., Song, Le, et al.
(2021). Universal machine learning for topology optimization. Comput. Methods Appl.
Mech. Eng. 375, 112739. doi:10.1016/j.cma.2019.112739
Chisholm, D. (1967). A theoretical basis for the lockhart-martinelli correlation for
two-phase ﬂ ow. Int. J. Heat Mass Transf. 10 (12), 1767 – 1778. doi:10.1016/0017-
9310(67)90047-6
Chokphoemphun, S., Somporn, H., Thongdaeng, S., and Chokphoemphun, S. (2020).
Experimental study and neural networks prediction on thermal performance
assessment of grooved channel air heater. Int. J. Heat Mass Transf. 163, 120397.
December. doi:10.1016/j.ijheatmasstransfer.2020.120397
Clark, L. A., and Pregibon, D. (2017). “ Tree-based models, ” in Statistical models in S
(England, UK: Routledge), 377 – 419.
Cunningham, P., Cord, M., and Jane Delany, S. (2008). “ Supervised learning, ” in
Machine learning techniques for multimedia (Cham: Springer), 21 – 49. doi:10.1007/978-
3-540-75171-7_2
Dalkilic, A. S., Çebi, A., and Celen, A. (2019). Numerical analyses on the prediction of
Nusselt numbers for upward and downward ﬂ ows of water in a smooth pipe: effects of
buoyancy and property variations. J. Therm. Eng. 5, 166 – 180. Yildiz Technical
University Press. doi:10.18186/thermal.540367
Dietterich, T. (1995). Over ﬁ tting and undercomputing in machine learning. ACM
Comput. Surv. (CSUR) 27 (3), 326 – 327. doi:10.1145/212094.212114
Dobbelaere, M. R., Plehiers, P. P., Van de Vijver, R., Stevens, C. V., and Van Geem, K.
M. (2021). Machine learning in chemical engineering: strengths, weaknesses,
opportunities, and threats. Engineering 7 (9), 1201 – 1211. doi:10.1016/j.eng.2021.03.019
Du, R., Zou, J., An, J., and Huang, L. (2023). A regression-based approach for the
explicit modeling of simultaneous heat and mass transfer of air-to-refrigerant
microchannel heat exchangers. Appl. Therm. Eng. 235, 121366. Available at SSRN
4436121. doi:10.1016/j.applthermaleng.2023.121366
Du, X., Chen, Z., Qi, M., and Song, Y. (2020). Experimental analysis and ANN prediction on
performances of ﬁ nned oval-tube heat exchanger under different air inlet angles with limited
experimental data. Open Phys. 18 (1), 968 – 980. doi:10.1515/phys-2020-0212
Dy, J. G., and Brodley, C. E. (2000). “ Feature subset selection and order identi ﬁ cation
for unsupervised learning, ” in Icml (Citeseer), 247 – 254.
El-Amin, M. F., Alwated, B., and Hoteit, H. A. (2023). Machine learning prediction of
nanoparticle transport with two-phase ﬂ ow in porous media. Energies 16 (2), 678.
doi:10.3390/en16020678
El-Said, E. M., Abd Elaziz, M., and Elsheikh, A. H. (2021). Machine learning
algorithms
for
improving
the
prediction
of
air
injection
effect
on
the
thermohydraulic performance of shell and tube heat exchanger. Appl. Therm. Eng.
185, 116471. February. doi:10.1016/j.applthermaleng.2020.116471
Ewim, D. R. E., Okwu, M. O., Onyiriuka, E. J., Abiodun, A. S., Abolarin, S. M., and
Kaood, A. (2021). A quick review of the applications of arti ﬁ cial neural networks (ANN)
in the modelling of thermal systems. Eng. Appl. Sci. Res . Paulus Editora. doi:10.14456/
easr.2022.45
Fawaz, A., Hua, Y., Le Corre, S., Fan, Y., and Luo, L. (2022). Topology optimization of
heat exchangers: a review. Energy 252, 124053. doi:10.1016/j.energy.2022.124053
Feurer, M., and Hutter, F. (2019). Hyperparameter optimization. Autom. Mach.
Learn. Methods, Syst. Challenges , 3 – 33. doi:10.1007/978-3-030-05318-5_1
Foley, A. (2013). How to model a shell and tube heat exchanger. COMSOL Blog .
September 11, 2013.
Gao, N., Wang, X., Xuan, Y., and Chen, G. (2019). An arti ﬁ cial neural network for the
residual isobaric heat capacity of liquid HFC and HFO refrigerants. Int. J. Refrig. 98,
381 – 387. February. doi:10.1016/j.ijrefrig.2018.10.016
Garcia, J. J., Garcia, F., Bermúdez, J., and Machado, L. (2018). Prediction of pressure
drop during evaporation of R407C in horizontal tubes using arti ﬁ cial neural networks.
Int. J. Refrig. 85, 292 – 302. January. doi:10.1016/j.ijrefrig.2017.10.007
Gholizadeh, M., Jamei, M., Ahmadianfar, I., and Pourrajab, R. (2020). Prediction of
nano ﬂ uids viscosity using random forest (RF) approach. Chemom. Intelligent
Laboratory Syst. 201, 104010. doi:10.1016/j.chemolab.2020.104010
Giannetti, N., Redo, M. A., Jeong, J., Yamaguchi, S., Saito, K., Kim, H., et al. (2020).
Prediction of two-phase ﬂ ow distribution in microchannel heat exchangers using
arti ﬁ cial neural network. Int. J. Refrig. 111, 53 – 62. doi:10.1016/j.ijrefrig.2019.11.028
Giri Nandagopal, M. S., Abraham, E., and Selvaraju, N. (2017). Advanced neural
network prediction and system identi ﬁ cation of liquid-liquid ﬂ ow patterns in circular
microchannels with varying angle of con ﬂ uence. Chem. Eng. J. 309, 850 – 865. doi:10.
1016/j.cej.2016.10.106
Giri Nandagopal, M. S., and Selvaraju, N. (2016). Prediction of liquid – liquid ﬂ ow patterns
in a Y-junction circular microchannel using advanced neural network techniques. Industrial
Eng. Chem. Res. 55 (43), 11346 – 11362. doi:10.1021/acs.iecr.6b02438
Godfrey, N., Somtochukwu, B. K., Whidborne, J. F., and Rana, Z. (2021). Non-
intrusive classi ﬁ cation of gas-liquid ﬂ ow regimes in an S-shaped pipeline riser using a
Doppler ultrasonic sensor and deep neural networks. Chem. Eng. J. 403, 126401. doi:10.
1016/j.cej.2020.126401
Greenwell, B. M. (2017). Pdp: an R package for constructing partial dependence plots.
R. J. 9 (1), 421. doi:10.32614/rj-2017-016
Gupta, A. K., Kumar, P., Sahoo, R. K., Sahu, A. K., and Sarangi, S. K. (2017).
Performance measurement of plate ﬁ n heat exchanger by exploration: ANN, ANFIS, ga,
and sa. J. Comput. Des. Eng. 4 (1), 60 – 68. doi:10.1016/j.jcde.2016.07.002
Guyon, I., and Elisseeff, A. (2003). An introduction to variable and feature selection
andré elisseeff. J. Mach. Learn. Res. 3, 1157 – 1182. doi:10.1162/153244303322753616
Guyon, I., Gunn, S., Nikravesh, M., and Zadeh, L. A. (2008). Feature extraction:
foundations and applications. Vol. 207 . Cham: Springer.
Hall, S. (2012). “ 2 - heat exchangers, ” in Branan ’ s rules of thumb for chemical
engineers . Editor Stephen Hall. 5 (Oxford: Butterworth-Heinemann), 27 – 57. doi:10.
1016/B978-0-12-387785-7.00002-5
Hanus, R., Zych, M., Kusy, M., Jaszczur, M., and Petryka, L. (2018). Identi ﬁ cation of
liquid-gas ﬂ ow regime in a pipeline using gamma-ray absorption technique and
computational intelligence methods. Flow Meas. Instrum. 60, 17 – 23. doi:10.1016/j.
ﬂ owmeasinst.2018.02.008
Harpole, G. M., and Eninger, J. E. (1991). “ Micro-Channel heat exchanger
optimization, ” in Proceedings - IEEE Semiconductor Thermal and Temperature
Measurement Symposium, Phoenix, AZ, USA, 12-14 February 1991 (Publ by IEEE),
59 – 63. doi:10.1109/stherm.1991.152913
Hassan, H., Abdelrahman, S.M.-B., and Gonzálvez-Maciá, J. (2016). Two-
dimensional numerical modeling for the air-side of minichannel evaporators
accounting for partial dehumidi ﬁ cation scenarios and tube-to-tube heat conduction.
Int. J. Refrig. 67, 90 – 101. July. doi:10.1016/j.ijrefrig.2016.04.003
Hojjat, M. (2020). Nano ﬂ uids as coolant in a shell and tube heat exchanger: ANN
modeling and multi-objective optimization. Appl. Math. Comput. 365, 124710. doi:10.
1016/j.amc.2019.124710
Hop ﬁ eld, J. J. (1982). Neural networks and physical systems with emergent collective
computational abilities. Proc. Natl. Acad. Sci. U. S. A. 79, 2554 – 2558. doi:10.1073/pnas.79.8.2554
Hosseini, S., Khandakar, A., Chowdhury, M. E., Ayari, M. A., Rahman, T.,
Chowdhury, M. H., et al. (2022). Novel and robust machine learning approach for
estimating the fouling factor in heat exchangers. Energy Rep. 8, 8767 – 8776. November.
doi:10.1016/j.egyr.2022.06.123
Hughes, M. T., Chen, S. M., and Garimella, S. (2022). Machine-learning-based heat
transfer and pressure drop model for internal ﬂ ow condensation of binary mixtures. Int.
J. Heat Mass Transf. 194, 123109. September. doi:10.1016/j.ijheatmasstransfer.2022.
123109
Jabbar, H., and Khan, R. Z. (2015). Methods to avoid over- ﬁ tting and under- ﬁ tting in
supervised machine learning (comparative study). Comput. Sci. Commun. Instrum.
Devices 70. doi:10.3850/978-981-09-5247-1_017
Jagielski, M., Carlini, N., Berthelot, D., Kurakin, A., and Papernot, N. (2020). “ High
accuracy and high ﬁ delity extraction of neural networks, ” in Proceedings of the 29th
USENIX conference on security symposium , 1345 – 1362.
Frontiers in Energy Research
frontiersin.org
21
Zou et al.
10.3389/fenrg.2023.1294531

---

Jensen, D. D., and Cohen, P. R. (2000). Multiple comparisons in induction algorithms.
Mach. Learn. 38 (3), 309 – 338. doi:10.1023/A:1007631014630
Khan, U., Pao, W., Sallih, N., and Hassan, F. (2022). Flow regime identi ﬁ cation in gas-
liquid two-phase ﬂ ow in horizontal pipe by deep learning. J. Adv. Res. Appl. Sci. Eng.
Technol. 27 (1), 86 – 91. doi:10.37934/araset.27.1.8691
Kramer, O. (2013). Dimensionality reduction with unsupervised nearest neighbors
123. Intell. Syst. Ref. Libr. 51. doi:10.1007/978-3-642-38652-7
Krishnayatra, G., Tokas, S., and Kumar, R. (2020). Numerical heat transfer analysis
and predicting thermal performance of ﬁ ns for a novel heat exchanger using machine
learning. Case Stud. Therm. Eng. 21, 100706. October. doi:10.1016/j.csite.2020.100706
Kumar, P. C. M., and Rajappa, B. (2020). A review on prediction of thermo physical
properties of heat transfer nano ﬂ uids using intelligent techniques. Mater. Today Proc.
21, 415 – 418. doi:10.1016/j.matpr.2019.06.379
Kunjuraman, S., and Velusamy, B. (2021). Performance evaluation of shell and tube
heat exchanger through ANN and ANFIS model for dye recovery from textile ef ﬂ uents.
Energy Sources, Part A Recovery, Util. Environ. Eff. 43 (13), 1600 – 1619. doi:10.1080/
15567036.2020.1832627
Kuzucanl ı , S. A., Vatansever, C., Ya ş ar, A. E., and Karadeniz, Z. H. (2022). Assessment
of fouling in Plate Heat exchangers using classi ﬁ cation machine learning algorithms .
doi:10.34641/clima.2022.127
Kwon, B., Ejaz, F., and LeslieHwang, K. (2020). Machine learning for heat transfer
correlations. Int. Commun. Heat Mass Transf. 116, 104694. July. doi:10.1016/j.
icheatmasstransfer.2020.104694
Li, Q., Qi, Z., Yu, S., Sun, J., and Cai, W. (2023). Study on thermal-hydraulic
performance of printed circuit heat exchangers with supercritical methane based on
machine learning methods. Energy 282, 128711. doi:10.1016/j.energy.2023.128711
Li, Z., Aute, V., and Ling, J. (2019). Tube- ﬁ n heat exchanger circuitry optimization
using integer permutation based genetic algorithm. Int. J. Refrig. 103, 135 – 144. doi:10.
1016/j.ijrefrig.2019.04.006
Li, Ze Yu, Shao, L. L., and Zhang, C.Lu (2016). Modeling of ﬁ nned-tube evaporator
using neural network and Response surface methodology. J. Heat Transf. 138 (5). doi:10.
1115/1.4032358
Lindqvist, K., Wilson, Z. T., Næss, E., and Sahinidis, N. V. (2018). A machine learning
approach to correlation development applied to ﬁ n-tube bundle heat exchangers.
Energies 11 (12), 3450. doi:10.3390/en11123450
Longo, G. A., Giulia, R., Claudio, Z., Ludovico, O., Mauro, Z., and Brown, J. S. (2020c).
“ Application of an arti ﬁ cial neural network (ANN) for predicting low-GWP refrigerant
condensation heat transfer inside herringbone-type brazed Plate Heat exchangers
(BPHE) ” .
Int.
J.
Heat
Mass
Transf.
156,
119824.
August.
doi:10.1016/j.
ijheatmasstransfer.2020.119824
Longo, G. A., Simone, M., Giulia, R., Claudio, Z., Ludovico, O., and Mauro, Z. (2020b).
Application of an arti ﬁ cial neural network (ANN) for predicting low-GWP refrigerant boiling
heat transfer inside brazed Plate Heat exchangers (BPHE). Int. J. Heat Mass Transf. 160,
120204. October. doi:10.1016/j.ijheatmasstransfer.2020.120204
Longo, G. A., Mancin, S., Righetti, G., Zilio, C., Ceccato, R., and Salmaso, L. (2020a).
Machine learning approach for predicting refrigerant two-phase pressure drop inside
brazed Plate Heat exchangers (BPHE). Int. J. Heat Mass Transf. 163, 120450. December.
doi:10.1016/j.ijheatmasstransfer.2020.120450
Ma, T., Guo, Z., Lin, M., and Wang, Q. (2021). Recent trends on nano ﬂ uid heat
transfer machine learning research applied to renewable energy. Renew. Sustain. Energy
Rev. 138, 110494. doi:10.1016/j.rser.2020.110494
Ma, Y., Liu, C., Jiaqiang, E., Mao, X., and Yu, Z. (2022). Research on modeling and
parameter sensitivity of ﬂ ow and heat transfer process in typical rectangular
microchannels: from a data-driven perspective. Int. J. Therm. Sci. 172, 107356.
February. doi:10.1016/j.ijthermalsci.2021.107356
Mahesh, B. (2018). Machine learning algorithms-A review machine learning
algorithms-A review view Project self ﬂ owing generator view Project batta Mahesh
independent researcher machine learning algorithms-A review. Int. J. Sci. Res . doi:10.
21275/ART20203995
Maleki, A., Haghighi, A, and Mahariq, I. (2021). Machine learning-based approaches
for modeling thermophysical properties of hybrid nano ﬂ uids: a comprehensive review.
J. Mol. Liq. 322, 114843. doi:10.1016/j.molliq.2020.114843
Mangalathu, S., Hwang, S. H., and Jeon, J. S. (2020). Failure mode and effects analysis of RC
members based on machine-learning-based SHapley additive ExPlanations (SHAP) approach.
Eng. Struct. 219, 110927. doi:10.1016/j.engstruct.2020.110927
Meghdadi Isfahani, A. H., Reiszadeh, M., Yaghoubi Koupaye, S., and Honarmand, M.
(2017). Empirical correlations and an arti ﬁ cial neural network approach to estimate
saturated vapor pressure of refrigerants. Phys. Chem. Res. 5 (2), 281 – 292. doi:10.22036/
pcr.2017.41111
Michalski, R. S. (2000). Learnable evolution model: evolutionary processes guided by
machine learning. Mach. Learn. 38 (1), 9 – 40. doi:10.1023/A:1007677805582
Michalski, R. S., and Kaufman, K. A. (2006). Intelligent evolutionary design: a new
approach to optimizing complex engineering systems and its application to designing
heat exchangers. Int. J. Intelligent Syst. 21 (12), 1217 – 1248. doi:10.1002/int.20182
Mirzaei, M., Hassan, H., and Fadakar, H. (2017). Multi-objective optimization of
shell-and-tube heat exchanger by constructal theory. Appl. Therm. Eng. 125, 9 – 19.
doi:10.1016/j.applthermaleng.2017.06.137
Mohanty, D. K. (2017). Application of neural network model for predicting fouling
behaviour of a shell and tube heat exchanger. Int. J. Industrial Syst. Eng. 26, 228. doi:10.
1504/IJISE.2017.083674
Montañez-Barrera, J. A., Barroso-Maldonado, J. M., Bedoya-Santacruz, A. F., and
Mota-Babiloni, A. (2022). Correlated-informed neural networks: a new machine
learning framework to predict pressure drop in micro-channels. Int. J. Heat Mass
Transf. 194, 123017. September. doi:10.1016/j.ijheatmasstransfer.2022.123017
Moradkhani, M. A., Hosseini, S. H., and Karami, M. (2022a). Forecasting of saturated
boiling heat transfer inside smooth helically coiled tubes using conventional and
machine learning techniques. Int. J. Refrig. 143, 78 – 93. November. doi:10.1016/j.
ijrefrig.2022.06.036
Moradkhani, M. A., Hosseini, S. H., and Song, M. (2022b). Robust and general
predictive models for condensation heat transfer inside conventional and mini/micro
channel heat exchangers. Appl. Therm. Eng. 201, 117737. January. doi:10.1016/j.
applthermaleng.2021.117737
Moukalled, F., Mangani, L., and Darwish, M. (2016). Fluid mechanics and its
applications the ﬁ nite volume method in computational ﬂ uid dynamics . Cham:
Springer. Available at: http://www.springer.com/series/5980 .
Müller-Steinhagen, H. (1999). “ Cooling-water fouling in heat exchangers, ” in
Advances in heat transfer . Editors P. H. James, T. F. Irvine, Y. I. Cho, and
G. A. Greene (Amsterdam, Netherlands: Elsevier), 33, 415 – 496. doi:10.1016/S0065-
2717(08)70307-1
Muthukrishnan, S., Krishnaswamy, H., Thanikodi, S., Sundaresan, D., and
Venkatraman, V. (2020). Support vector machine for modelling and simulation of
heat exchangers. Therm. Sci. 24, 499 – 503. 1PartB. doi:10.2298/TSCI190419398M
Nabipour, M. (2018). Prediction of surface tension of binary refrigerant mixtures
using arti ﬁ cial neural networks. Fluid Phase Equilibria 456, 151 – 160. January. doi:10.
1016/j. ﬂ uid.2017.10.020
Naja ﬁ , B., Ardam, K., Hanu š ovský, A., Rinaldi, F., and Luigi Pietro Maria Colombo,
(2021). Machine learning based models for pressure drop estimation of two-phase
adiabatic air-water ﬂ ow in micro- ﬁ nned tubes: determination of the most promising
dimensionless feature set. Chem. Eng. Res. Des. 167, 252 – 267. March. doi:10.1016/j.
cherd.2021.01.002
Naphon, P., Wiriyasart, S., Arisariyawong, T., and Nakharintr, L. (2019). ANN,
numerical and experimental analysis on the jet impingement nano ﬂ uids ﬂ ow and heat
transfer characteristics in the micro-channel heat sink. Int. J. Heat Mass Transf. 131,
329 – 340. doi:10.1016/j.ijheatmasstransfer.2018.11.073
Onsager, L. (1931). Reciprocal relations in irreversible processes. I. Phys. Rev. 37 (4),
405 – 426. doi:10.1103/PhysRev.37.405
Paris, G., Robilliard, D., and Fonlupt, C. (2003). “ Exploring over ﬁ tting in genetic
programming, ” in International conference on arti ﬁ cial evolution (evolution arti ﬁ cielle)
(Cham: Springer), 267 – 277. doi:10.1007/978-3-540-24621-3_22
Patil, M. S., Seo, J. H., and Moo Yeon Lee (2017). “ Heat transfer characteristics of the
heat exchangers for refrigeration, ” in Air conditioning and heap pump systems under
frosting, defrosting and dry/wet conditions — a review. ” applied thermal engineering
(Amsterdam, Netherlands: Elsevier Ltd). doi:10.1016/j.applthermaleng.2016.11.107
Peng, H., and Xiang, L. (2015). Predicting thermal-hydraulic performances in
compact heat exchangers by support vector regression. Int. J. Heat Mass Transf. 84,
203 – 213. doi:10.1016/j.ijheatmasstransfer.2015.01.017
Pourkiaei, S. M., Hossein Ahmadi, M., and Mahmoud Hasheminejad, S. (2016).
Modeling and experimental veri ﬁ cation of a 25W fabricated PEM fuel cell by parametric
and GMDH-type neural network. Mech. Industry 17 (1), 105. doi:10.1051/meca/
2015050
Prigogine, I., and Van Rysselberghe, P. (1963). Introduction to thermodynamics of
irreversible processes. J. Electrochem. Soc. 110 (4), 97C. doi:10.1149/1.2425756
Prithiviraj, M., and Andrews, M. J. (1998). Three dimensional numerical simulation
of shell-and-tube heat exchangers. Part I: foundation and ﬂ uid mechanics. Numer. Heat.
Transf. Part A Appl. 33 (8), 799 – 816. doi:10.1080/10407789808913967
Qiu, Y., Garg, D., Sung, M. K., Mudawar, I., and ChiragKharangate, R. (2021).
Machine learning algorithms to predict ﬂ ow boiling pressure drop in mini/micro-
channels based on universal consolidated data. Int. J. Heat Mass Transf. 178, 121607.
October. doi:10.1016/j.ijheatmasstransfer.2021.121607
Ramezanizadeh, M., Hossein Ahmadi, M., Alhuyi Nazari, M., Sadeghzadeh, M., and
Chen, L. (2019). “ A review on the utilized machine learning approaches for modeling
the dynamic viscosity of nano ﬂ uids, ” in Renewable and sustainable energy reviews
(Amsterdam, Netherlands: Elsevier Ltd). doi:10.1016/j.rser.2019.109345
Rattan, P., Penrice, D. D., and Simonetto, D. A. (2022). Arti ﬁ cial intelligence and
machine learning: what you always wanted to know but were afraid to ask. Gastro Hep
Adv. 1 (1), 70 – 78. doi:10.1016/j.gastha.2021.11.001
Roshani, G. H., Nazemi, E., and Roshani, M. M. (2017). A novel method for ﬂ ow
pattern identi ﬁ cation in unstable operational conditions using gamma ray and radial
basis function. Appl. Radiat. Isotopes 123, 60 – 68. doi:10.1016/j.apradiso.2017.02.023
Frontiers in Energy Research
frontiersin.org
22
Zou et al.
10.3389/fenrg.2023.1294531

---

Roy, U., and Majumder, M. (2019). Evaluating heat transfer analysis in heat
exchanger using NN with IGWO algorithm. Vacuum 161, 186 – 193. March. doi:10.
1016/j.vacuum.2018.12.042
Shannak, B. A. (2008). Frictional pressure drop of gas liquid two-phase ﬂ ow in pipes.
Nucl. Eng. Des. 238 (12), 3277 – 3284. doi:10.1016/j.nucengdes.2008.08.015
Shen, C., Zheng, Q., Shang, M., Zha, Li, and Su, Y. (2020). Using deep learning to
recognize liquid – liquid ﬂ ow patterns in microchannels. AIChE J. 66 (8), e16260. doi:10.
1002/aic.16260
Shojaeefard, M. H., Zare, J., Tabatabaei, A., and Mohammadbeigi, H. (2017).
Evaluating different types of arti ﬁ cial neural network structures for performance
prediction of compact heat exchanger. Neural Comput. Appl. 28 (12), 3953 – 3965.
doi:10.1007/s00521-016-2302-z
Singh, A., Sahu, D., and Verma, Om P. (2022). Study on performance of working
model of heat exchangers. Mater. Today Proc. Sept . doi:10.1016/j.matpr.2022.
09.373
Skrypnik, A. N., Shchelchkov, A. V., Gortyshov, Yu F., and Popov, I. A. (2022).
Arti ﬁ cial neural networks application on friction factor and heat transfer coef ﬁ cients
prediction in tubes with inner helical- ﬁ nning. Appl. Therm. Eng. 206, 118049. April.
doi:10.1016/j.applthermaleng.2022.118049
Sosnovik,
I.,
and
Oseledets,
I.
(2019).
Neural
networks
for
topology
optimization. Russ. J. Numer. Analysis Math. Model. 34 (4), 215 – 223. doi:10.
1515/rnam-2019-0018
Subbappa, R., Aute, V., and Ling, J. (2022). Development and comparative evaluation
of machine learning algorithms for performance approximation of air-to-refrigerant heat
exchangers .
Sun, L., Zhang, Y., Zheng, X., Yang, S., and Qin, Y. (2008). “ Research on the fouling
prediction of heat exchanger based on support vector machine, ” in Proceedings -
International Conference on Intelligent Computation Technology and Automation,
ICICTA 2008, Changsha, Hunan, 20-22 October 2008, 240 – 244. 1. doi:10.1109/
ICICTA.2008.156
Sundar, S., Rajagopal, M. C., Zhao, H., Kuntumalla, G., Meng, Y., Chang, Ho C., et al.
(2020). Fouling modeling and prediction approach for heat exchangers using deep
learning.
Int.
J.
Heat
Mass
Transf.
159,
120112.
October.
doi:10.1016/j.
ijheatmasstransfer.2020.120112
Thibault, J., and Grandjean, B. P. A. (1991). A neural network methodology for heat
transfer data analysis. Int. J. Heat Mass Transf. 34, 2063 – 2070. doi:10.1016/0017-
9310(91)90217-3
Thombre, M., Mdoe, Z., and Johannes, J. (2020). Data-driven robust optimal
operation of thermal energy storage in industrial clusters. Processes 8 (2), 194.
doi:10.3390/pr8020194
Uguz, S., and Osman, I. (2022). Prediction of the parameters affecting the performance of
compact heat exchangers with an innovative design using machine learning techniques.
J. Intelligent Manuf. 33 (5), 1393 – 1417. doi:10.1007/s10845-020-01729-0
Vellido, A., José David Martín-Guerrero, and Lisboa, P. J. G. (2012). “ Making
machine learning models interpretable, ” in ESANN, 12 (Belgium: Bruges),
163 – 172.
Wang, Bo, and Wang, J. (2021). “ Application of arti ﬁ cial intelligence in computational
ﬂ uid
dynamics. ”
industrial
and
engineering
chemistry
research .
Washington,
United States: American Chemical Society. doi:10.1021/acs.iecr.0c05045
Wang, Q., Xie, G., Zeng, M., and Luo, L. (2006). Prediction of heat transfer rates for
shell-and-tube heat exchangers by arti ﬁ cial neural networks approach. J. Therm. Sci 15,
257 – 262. doi:10.1007/s11630-006-0257-6
Wang, X., Li, Y., Yan, Y., Wright, E., Gao, N., and Chen, G. (2020). Prediction on the
viscosity and thermal conductivity of hfc/hfo refrigerants with arti ﬁ cial neural network
models. Int. J. Refrig. 119, 316 – 325. November. doi:10.1016/j.ijrefrig.2020.07.006
Wiering, M. A., and Van Otterlo, M. (2012). Reinforcement learning. Adapt. Learn.
Optim. 12 (3), 729. doi:10.1007/978-3-642-27645-3
Xie, C., Yan, G., Ma, Q., Elmasry, Y., Singh, P. K., Algelany, A. M., et al. (2022a). Flow
and heat transfer optimization of a ﬁ n-tube heat exchanger with vortex generators using
Response surface methodology and arti ﬁ cial neural network. Case Stud. Therm. Eng. 39,
102445. November. doi:10.1016/j.csite.2022.102445
Xie, X., Samaei, A., Guo, J., Liu, W. K., and Gan, Z. (2022b). Data-driven discovery of
dimensionless numbers and governing laws from scarce measurements. Nat. Commun.
13 (1), 7562. doi:10.1038/s41467-022-35084-w
Yang, K. T. (2008). Arti ﬁ cial neural networks (ANNs): a new paradigm for thermal
science and engineering. J. Heat Transf. 130 (9). doi:10.1115/1.2944238
Zendehboudi, A., and Li, X. (2017). A robust predictive technique for the pressure
drop during condensation in inclined smooth tubes. Int. Commun. Heat Mass Transf.
86, 166 – 173. August. doi:10.1016/j.icheatmasstransfer.2017.05.030
Zhang, G., Wang, B., Li, X., Shi, W., and Cao, Y. (2019a). Review of experimentation
and modeling of heat and mass transfer performance of ﬁ n-and-tube heat exchangers
with dehumidi ﬁ cation. Appl. Therm. Eng 146, 701 – 717. Elsevier Ltd. doi:10.1016/j.
applthermaleng.2018.10.032
Zhang, Xu, Zou, Y., Li, S., and Xu, S. 2019b. “ A weighted auto regressive LSTM based
approach for chemical processes modeling. ” Neurocomputing 367: 64 – 74. doi:10.1016/j.
neucom.2019.08.006
Zheng, X., Yang, R., Wang, Q., Yan, Y., Zhang, Yu, Fu, J., et al. (2022). Comparison of
GRNN and RF algorithms for predicting heat transfer coef ﬁ cient in heat exchange
channels with bulges. Appl. Therm. Eng. 217, 119263. November. doi:10.1016/j.
applthermaleng.2022.119263
Zhi, L. H., Hu, P., Chen, L. X., and Zhao, G. (2018). Viscosity prediction for six pure
refrigerants using different arti ﬁ cial neural networks. Int. J. Refrig. 88, 432 – 440. doi:10.
1016/j.ijrefrig.2018.02.011
Zhou, L., Garg, D., Qiu, Y., Sung, M. K., Mudawar, I., and Chirag Kharangate, R.
(2020a). Machine learning algorithms to predict ﬂ ow condensation heat transfer
coef ﬁ cient in mini/micro-channel utilizing universal data. Int. J. Heat Mass Transf.
162, 120351. December. doi:10.1016/j.ijheatmasstransfer.2020.120351
Zhou, X., Hu, Y., Liang, W., Ma, J., and Jin, Q. (2020b). Variational LSTM enhanced
anomaly detection for industrial big data. IEEE Trans. Industrial Inf. 17 (5), 3469 – 3477.
doi:10.1109/TII.2020.3022432
Zhu, G., Wen, T., and Zhang, D. (2021a). Machine learning based approach for the
prediction of ﬂ ow boiling/condensation heat transfer performance in mini channels
with
serrated
ﬁ ns.
Int.
J.
Heat
Mass
Transf.
166,
120783.
doi:10.1016/j.
ijheatmasstransfer.2020.120783
Zhu, X., and Goldberg, A. B. (2009). Introduction to semi-supervised learning.
Synthesis Lect. Artif. Intell. Mach. Learn. 3 (1), 1 – 130. doi:10.1007/978-3-031-01548-9
Zhu, X. J. (2005). Semi-supervised learning literature survey . Available at: http://
digital.library.wisc.edu/1793/60444 .
Zhu, Z., Li, J., Peng, H., and Liu, D. (2021b). Nature-inspired structures applied in
heat transfer enhancement and drag reduction. Micromachines 12 (6), 656. doi:10.3390/
mi12060656
Zolfaghari, H., and Youse ﬁ , F. (2017). Thermodynamic properties of lubricant/
refrigerant mixtures using statistical mechanics and arti ﬁ cial intelligence. Int.
J. Refrig. 80, 130 – 144. August. doi:10.1016/j.ijrefrig.2017.04.025
Frontiers in Energy Research
frontiersin.org
23
Zou et al.
10.3389/fenrg.2023.1294531

---

### Nomenclature
ANFIS
Adaptive Neuro Fuzzy Interface System
ANN
Arti ﬁ cial Neural Network
ANN − FF
ANN-Function Fitting
ANN − PR
ANN-Pattern Recognition
CFN
Cascade Forward Network
CoINN
Correlated-Informed Neural Networks
DT
Decision Tree
FEM
Finite Element Method
FFBN
Feed Forward Back Propagation Network
FFNN
Feed-Forward Neural Network
FVM
Finite Volume Method
GA − PLCIS
Genetic Algorithm-power Law Committee with Intelligent Systems
GA − LSSVM
Genetic Algorithm-least Square Support Vector Machine
GBM
Gradient Boosting Machine
GBT
Gradient Boosting Tree
GPR
Gaussian Process Regression
GRNN
General Regression Neural Network
HRBF
Hybrid Radial Basis Function
KNN
K-Nearest Neighbor
PNN
Probabilistic Neural Network
PSO − ANN
Particle Swam Optimization-Arti ﬁ cial Neural Network
RBF
Radial Basis Function
RF
Random Forest
RR
Ridge Regression
SVM
Support Vector Machine
SVR
Support Vector Regression
A
Cross-sectional area
atm
Atmosphere
Bd
Bond number, Bd 
g ( ρ f − ρ g ) D 2 h
σ
Bo
Boiling number, Bo 
q ″
Δ HG
BPHEs
Brazed Plate Heat Exchangers
Ca
Capillary number
CF
Condensate ﬂ ow
Co
Convection number, Co  ( 1 − x
x ) 0 . 8 (
ρ g
ρ f ) 0 . 5
CT
Condensate temperature
D
Diameter of ﬂ ow channel
Dc
Coil diameter
Dh
Hydraulic diameter of ﬂ ow channel
Dt
Tube diameter
e
Mean ﬁ n height
f
Friction factor (by Haaland) ( Hall, 2012 )
f o
Isothermal friction factor in forced convection
FIT
Feed inlet temperature
Fr
Froude number
f vp
Friction factor, Parlatan et al. ‘ s friction factor
g
Gravity acceleration
G
Mass ﬂ ux
Ga
Galileo number, Ga 
ρ f g ( ρ f − ρ g ) D 3 h
μ 2
f
Gr
Grashof number
h
Enthalpy
H
Heat of vaporization of the ﬂ uid
HTC
Heat transfer coef ﬁ cient
IA
Inclination angle
j
Colburn factor
Ja
Jakob number, L ( T sat − T wall ) Cp
h lv
Ka
Kapitza number, Ka 
μ 4
f g
ρ f σ 3
L
Length of ﬁ n
_ m
Mass ﬂ ow rate
M
Mole molecule mass
n
Number of ﬁ ns
N
Number of channels
Ntr
Number of tube rows
Ntp
Number of tube-passes
Nu
Nusselt numbers
Nuo
Nusselt number in forced convection
P
Pressure
Δ P
Pressure drop
P 1
Pressure at cold inlet
P 2
Pressure at hot inlet
Pc
Critical pressure
Pcd
Cold ﬂ uid Pressure drop
Pcr
Critical pressure of the corresponding hydrocarbon to the
refrigerant
Pe
Peclet number
Phd
Hot ﬂ uid Pressure drop
Pr
Prandtl number
Pr g
Saturated vapor Prandtl number, Pr g 
μ g · Cp g
k g
q
Data point vector
q ″
Heat ﬂ ux
Q
Flow rate, liters/min
q ″ H
Heat ﬂ ux based on heated perimeter of channel
Re
Reynolds number
Frontiers in Energy Research
frontiersin.org
24
Zou et al.
10.3389/fenrg.2023.1294531

---

Res
Super ﬁ cial Reynolds numbers
Rh
Relative humidity
ScL
Liquid-phase Schmidt number
ScV
Vapor-phase Schmidt number
SF
Steam ﬂ ow
Suf
Saturated liquid Suratman number, Su f 
σ · ρ f · Dh
μ 2
f
Sug
Saturated vapor Suratman number, Su f 
σ · ρ g · Dh
μ 2
f
T
Temperature
T *
Dimensionless temperature glide
T 1
Temperature at inlet of cold ﬂ uid
T 2
Temperature at outlet of cold ﬂ uid
T 3
Temperature at inlet of hot ﬂ uid
T 4
Temperature at outlet of hot ﬂ uid
Tr
Reduced temperature
V
Volume ﬂ ow rate, m 3 /s
WPP
Pumping power
We
Weber number
x
Average volume quality
X
Lockhart-Martinelli parameter X  m l
m g

ρ g
ρ l

xv
Average volume quality
Greek symbols
α
Heat transfer coef ﬁ cient/aspect ratio α /arc angle
β
Inclination angle of the corrugation/attack angle
Δ
The differences
Φ
Enlargement factor of the corrugation
Γ
Take-off ratio
μ
Dynamic viscosity
η
Thermal enhancement factor
θ
Angle
ρ
Density
ω
Acentric factor
σ
Surface tension
Subscripts
a
Air
amb
Ambient
avg
Average
b
Bed
c
Critical
cw
Cold water
db
Dry bulb
eq
Equivalent
evap
Evaporator
f
Saturated liquid, ﬂ uid
f lue
Flue gas
f o
Liquid only
f ric
Frictional
g
Saturated vapor
go
Vapor only
hw
Hot water
i
Inlet
in
Inlet
int
Internal
l
Liquid
max
Max value
mix
Non-azeotropic mixtures
o
Outlet
r
Reduced
red
Reduced
ref
Refrigerant
s
Saturation
sp
Single-phase ﬂ ow
sup
Vapor super-heating
sys
System
tp
Two-phase ﬂ ow
v
Vapor
w
Water
wb
Wet bulb
wo
Water only
MAE
Mean Absolute Error
MedAE
Median Absolute Error
MRE
Mean Relative Error
RMSE
Root-Mean-Square Error
R 2
Coef ﬁ cient of Determination
Frontiers in Energy Research
frontiersin.org
25
Zou et al.
10.3389/fenrg.2023.1294531