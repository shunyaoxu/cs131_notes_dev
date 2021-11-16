---
title: Clustering
keywords: K-means, Mean-Shift, Clustering
order: 14
---

- [K-means clustering](#k-means-clustering)
	- [Motivation](#motivation)
	    - [Image Segmentation](#image_segmentation)
	- [Workflow](#workflow)
	    - [Clustering](#clustering)
	    - [K-Means Clustering](#Kclustering)
	    - [Segmentation Clustering](#Sclustering)
	    - [Updated K-Means](#UKM)
	- [Analysis](#analysis)
	    - [Feature Space](#fs)
	    - [Results of K-Means Clustering](#RKM)
	    - [Evaluating Clusters](#eclusters)
	    - [Choosing the Number of Clusters](#CtNoC)
	    - [Pros & Cons](#PaC)
- [Mean-Shift Clustering](#mean-shift-clustering)
    - [Mean-Shift Algorithm](#mean-shift-algorithm)
    - [Mean-Shift Clustering/Segmentation](#mean-shift-clustering/segmentation)
    - [Speed Up Mean-Shift](#mean-shift-algorithm)
        - [Computational Complexity Problem](#computational-complexity-problem)
        - [Speedup Methodology](#speedup-methodology)
    - [Technical Details](#technical-details)
    - [Summary Mean-Shift](#summary-mean-shift)

[//]: # (This is how you can make a comment that won't appear in the web page! It might be visible on some machines/browsers so use this only for development.)

[//]: # (Notice in the table of contents that [First Big Topic] matches #first-big-topic, except for all lowercase and spaces are replaced with dashes. This is important so that the table of contents links properly to the sections)

[//]: # (Leave this line here, but you can replace the name field with anything! It's used in the HTML structure of the page but isn't visible to users)

<a name='K-means clustering'></a>
## K-means clustering


### Background

This part of the class is focused on segments, looking at segmentation and clustering. Image segmentation has the goal of identifying groups of pixels that go together.

We have discussed Gestalt theory, whether we see objects as a whole or in parts. The whole will always be greater that the sum of its parts. Each part additionally has relationships that can yield new properties or features.

Proximity, similarity, common fate, and common region are a few factors that can be used to group elements.

This lecture is focused on K-means clustering and mean-shift clustering

### Motivation
#### Image Segmentation
Intensities in the image define the three groups that we have. Consequently, every pixel in the image can be labeled by the intensity category it falls into. For eample, we can segment the image by using the intensity feature. This is more difficult for more complicated images.

When the images are more complicated, we need to use clustering as a method to group the pixels of the image. Our goal for clustering in this case is to choose three "centers" as representative intensities, labeling each pixel based on which center it is closest to. We pick the cluster "centers" by minimizing the sum of square distance (SSD) between all of the points and their nearest cluster "center" $c_i$.

$$
SSD = \sum_{cluster i} \sum_{x \in cluster i}(x - c_i)^2
$$

### Workflow
#### Clustering
Clustarizing can also be used for summarization. In this case, our goal is to minimize the variance in data given clusters and to preserve information.
In this case, our formula is
$$
c^*, \delta^* = argmin_{c, \delta} \frac{1}{N} \sum_{j}^{N}\sum_{i}^{K}\delta_{ij} (c_i - x_j)^2
$$
where $c_i$ is the clustering center and $x_j$ is the data


Clustering can be seen as a "chicken and egg problem." If we knew cluster centers, we could easily allocate points to each group using our formulas to find the closest center to each point. However, if we knew group memberships, we could get the centers by solving for the mean in each group.

The problem we face is how to do both.

#### K-Means Clustering

K-means clustering is the solution to this. The steps to K-means clustering are as follows.
1. Initialize (t=0) : cluster centers $c_1$,...,$c_k$
2. Compute sigma^t: assign each point to the closest center

$\delta^t$ represents the set of assignment for each $x_j$ cluster $c_i$ at iteration t. The formula for $\delta^t$ is as follows:

$$\delta^t = argmin_{\delta} \frac{1}{N} \sum_{j}^{N} \sum_{i}^{K} \delta_{ij}^{t-1} (c_{i}^{t-1} - x_j)^2$$

We can measure the distance using euclidean, cosine, and other methods.

3. Computer $c^t$: update cluster centers as the mean of the points
The formula for $c^t$ is as follows:

$$c^t = argmin_{c} \frac{1}{N} \sum_{j}^{N} \sum_{i}^{K} \delta_{ij}^{t} (c_{i}^{t-1} - x_j)^2$$

4. Update $t = t + 1$, repeating steps 2-3 until stopped
$c^t$ no longer changes

![](https://i.imgur.com/BOkDIv6.png)




K-means clustering will converge to a local minimim solution. Generally, when data is spherical, k-means clustering results in a better fit because clusters will be of similar size. It is our job to choose the k value for our clustering, as in how many clusters to use for our data. The best value of k is dependent on the type of data that is being fitted.


#### Segmentation Clustering
Segmentation can also be seen as clustering.

![](https://i.imgur.com/d4k0jfB.png)


#### Updated K-Means

To prevent arbitrarily bad local minima, we can use an updated version of k-means. This method is as follows:
1. Randomly choose first center
2. Pick new center with probability proportional to $(x - c_i)^2$
3. Repeat until K centers

This method has an expected error of O(log K)

### Analysis

#### Feature Space

The feature space is an important component of how we group our pictures. Depending on what we choose for the feature space, pixels can be grouped in different ways, such as grouping pixels based on intensity similarity, color similarity, or texture similarity.

It is common to smooth out cluster assignments. Generally, without smoothing, there will be outliers in clusters. Smoothing can be a helpful solution to cope with this issue. One way to go about this is by grouping pixels in an image based on both intensity and position.


#### Results of K-Means Clustering
K-means clustering that is based on color intensity is practically vector quantization of the image attributes. These clusters do not need to be spatially coherent.

However, if we cluster based on color intensity and pixal position, this enforces significantly more spatial coherence.

#### Evaluating Clusters
There are multiple ways to view evaluating clusters.
1. We can evaluate clusters from a generative approach. Using this approach we view how well points are reconstructed from the clusters.
2. We can evaluate clusters from a discriminative approach. Using this approach we view how well clusters correspond to labels. This does not include unsupervised clustering, since we do not know the labels in this case.

#### Choosing the Number of Clusters
To pick the right number of cluster to use, the best method is to try various numbers of clusters using a validation set and view their performances. We can use a technique known as "knee/elbow finding" to discover the proper number of clusters to use by finding where the largest abrupt change is in the performance from the validation set.



#### Pros & Cons
K-means clustering has both advantages and disadvantages, as shown by the figure below. 


| Pros | Cons | 
| -------- | -------- | 
| Easy to implement   | Sensitive to outliers   | 
Efficient for simple data | Need to specify k which is unknow |
| Visually represents data | Slow for multidimensional data |
| | Prone to local minima |

One way to overcome the second con is through a graphing method called knee-finding. In this method the objective function is plotted against different k functions.  An abrupt change in the objective function is indicative of a potentially effective k value. 




<a name='Mean-Shift Clustering'></a>
## Mean-Shift Clustering 
This is another clustering algorithm which is widely used. At its core, this algorithm is an iterative mode search. There are four main steps:
### The Algorithm
1. Initialize the random seed and window W
2. Calculate the center of gravity (mean) of W: $$\sum_{x\in W}xH(x)$$
3. Shift the search window to the mean
4. Repeat step 2 until convergence 

In practice, this can be done with many initial windows. Based on whether the points converge at the same mean or different ones, it is clear whether they are a part of the same cluster. For context, a cluster is all of the data points in the attraction basin of a mode. The following image shows one way to visualize the algorithm:

![](https://i.imgur.com/cfV40WR.png)

In this instance the algorithm converges on 2 clusters: left and right.  


<a name='Mean-Shift Clustering/Segmentation'></a>
### Mean-Shift Clustering/Segmentation
To practically use this algorithm, you need the features of an image (color, gradients, text, etc).Then you can initialize windows at all individual pixel locations and perform a mean shift for each of these. Finally, you merge the windows that end up near the same peak and you get a cluster.  

<a name='Speed Up Mean-Shift'></a>
### Speed Up Mean-Shift
<a name='Computational Complexity Problem'></a>
#### Computational Complexity Problem
To implement mean-shift clustering, we need to shift many windows for different but very close to each other starting points, and we also need to compute the point covered by the window for every time the window is shifted. Especially when the window are large, then a lot of computations will be redundant. Therefore, computational complexity is treated as a main challenge of the mean-shift algorithm. 
<a name='Speedup Methodology'></a>
#### Speedup Methodology
A simple speedup can be applied to solve this problem. As shown in the image below, we can assign all points within a certain radius $r$ of the endpoint to the same cluster.
![](https://i.imgur.com/SRZs3i4.png =500x300)
This method can be further optimized by defining a second circle with radius $c$. When the window is moving from the staring point to the end point, all the points covered by this small circle will be automatically assigned to this cluster.
![](https://i.imgur.com/3KwLcb7.png =500x300)
By appling these two seepdups, even shifting a single window can assigned a bunch of pixel points to a cluster. This makes the mean-shift clustering algorithm much more efficient and runs faster.

<a name='Technical Details'></a>
### Technical Details
Dive into technical details of mean-shift clustering. First, given $n$ data points $x_i\in R^d$ the multivariate kernel density function is calculated by:
$$
\hat{f_K} = \frac{1}{nh^d} \sum_{i=1}^{n}K(\frac{x-x_i}{h})
$$
where $K(x)$ is the function of a radially symmetric kernel and $h$ defines the radius of the kernel.

To shift the kernel, we need to compute the gradient of this kernel density function, which is:
$$
\nabla\hat{f}(x) = \frac{2c_{k,d}}{nh^{d+2}} [\sum_{i=1}^{n}g(\|\frac{x-x_i}{h})\|^2] [\frac{\sum_{i=1}^{n}x_ig(\|\frac{x-x_i}{h})\|^2}{\sum_{i=1}^{n}g(\|\frac{x-x_i}{h})\|^2}-x]
$$
where $g(x)$ denotes the derivative of the selected kernel profile. Note that the last term is the mean-shift vector that points towards the direction of maximum density.

In practice, the mean-shift calculation above can be analyzed into three procedures:
1. Compute the mean-shift vector $m$: $$[\frac{\sum_{i=1}^{n}x_ig(\|\frac{x-x_i}{h})\|^2}{\sum_{i=1}^{n}g(\|\frac{x-x_i}{h})\|^2}-x]$$
2. Translate the density window: $$x^{t+1}_i = x^{t}_i + m(x^{t}_i)$$
3. Iterate steps 1 and 2 until converge: $$\nabla f(x_i) = 0$$



Additionally, there could be many different choices of the kernel function $K(x)$. Here is some examples:
1. Rectangular: $$\phi(x) = \begin{cases} 1 \quad &a\leq x\leq b \\ 0 \quad &else \end{cases}$$
2. Gaussian: $$\phi(x) = e^{-\frac{x_2}{2\sigma^2}}$$
3. Epanechnikov: $$\phi(x) = \begin{cases} \frac{3}{4}(1-x^2) \quad &if|x|\leq1 \\ 0 \quad &else \end{cases}$$

<a name='Summary Mean-Shift'></a>
### Summary Mean-Shift
Mean-shift clustering has many advantages. Firstly, mean-shift clustering is pretty general. It is an application-independent tool which allow us to change the kernel, the distance, and the feature representation. Moreover, mean-shift is model-free since it doesn't assume any pior shape on data clusters. Also, we could use mean-shift to find variable number of modes. Additionally, mean-shift is robust to outliers.

However, the diaadvantages of mean-shift is also obvious. Firstly, its outputs depends on window size, and window size selection is not trivial. Secondly, mean-shift algorithm is relatively computational expensive. In addition, it doesn't scale very well with dimension of feature space.
