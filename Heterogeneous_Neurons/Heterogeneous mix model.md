## Heterogeneous mix model

### 0. Setup

1. Assume we have 2 modalities on the input layer: 

$\sigma$ with $N$ neruons, $\eta$ with $M$ neurons.
    
2. we have $N_c$ neurons on cortical layer, each neurons in one of the following types:

- $p_1$ fraction neurons, purely selective to $\sigma$:

$$h_1 = J^\sigma_1\sigma$$

  - $p_2$ fraction neurons, purely selective to $\eta$:

$$h_2 = J^\eta_1\eta$$

   - $p_3$ fraction neurons, selective to mix of both:

$$h_3 = J^\sigma_2\sigma+ J^\eta_2\eta$$

$p_1 + p_2 + p_3  = 1$
    
3. The total matrix before non-linerity at cortical layer is:

$$h = (h_1,h_2,h_3) = (J^\sigma_1\sigma,J^\eta_1\eta,J^\sigma_2\sigma+ J^\eta_2\eta)$$

To make $h_{ij} \sim \mathcal{N}(0,1)$, we have to normalize connection matrix: 

$$J_1^\sigma \sim \mathcal{N}(0,\frac{1}{N}) \quad J_1^\eta \sim \mathcal{N}(0,\frac{1}{M})$$

$$J_2^\sigma \sim \mathcal{N}(0,\frac{1}{2N}) \quad J_2^\eta \sim \mathcal{N}(0,\frac{1}{2M})$$



4. Add a threshold T to introduce non-linearity

$$m  = \theta(h - T)$$



### 1. Rank 

<img src="/Users/liminhuan/Library/Application Support/typora-user-images/image-20200526000236029.png" alt="image-20200526000236029" style="zoom:50%;" />

Important Message:

1. Increase the fraction of fully mixed neurons **could increase the rank (capacity)**, but i would **soon saturate**, if you don't have so many indepedent samples.  ----->  so **no need to have so much neurons fully mixed** 
2. change of p1 / p2, will barely change the rank

<img src="/Users/liminhuan/Library/Application Support/typora-user-images/image-20200526004104967.png" alt="image-20200526004104967" style="zoom:50%;" />





###  2. $\Delta m$

**Theoretical part**

$$\bar{h}=\left(J_{1}^{\sigma} \bar{\sigma}, J_{1}^{\eta} \bar{\eta}, J_{2}^{\sigma} \bar{\sigma}+J_{2}^{\eta} \bar{\eta}\right)$$

$$h=\left(J_{1}^{\sigma}\sigma, J_{1}^{\eta} \eta, J_2^{\sigma} \sigma+J_{2}^{\eta} \eta\right)$$

$$\langle h \bar{h}\rangle= N_{c} p_{1} \frac{1}{N} \cdot(1-\Delta \sigma)+N_{c} p_{2} \frac{1}{M}(1-\Delta \eta) +N_{c} p_{3}\left[\frac{1}{2 N}(1-\Delta \sigma)+\frac{1}{2 M}(1-\Delta \eta)\right]$$

Set $N = M$: 

$$
\begin{aligned}
\Rightarrow & \frac{N_{c}}{N}\left\{p_{1}(1-\Delta \sigma)+p_{2}(1-\Delta \eta)+\frac{p_{3}}{2}(1-\Delta \sigma)+\frac{p_{3}}{2}(1-\Delta \eta)\right\} \\
&=\frac{N_{c}}{N}[1-(\underbrace{p_{1}+\frac{p_{3}}{2}}_{x})\Delta \sigma  - (\underbrace{p_2+\frac{p_3}{2}}_{1-x})\Delta\eta]\\
&=\frac{N_{c}}{N}[1-x \Delta \sigma-(1-x) \Delta \eta] \\
&=\frac{N_{c}}{N}[1-\Delta \eta+(\Delta \eta-\Delta \sigma) x]
\end{aligned}
$$

Set a benchmark for intuition, let's assume $\Delta \sigma > \Delta \eta$, so $\Delta \eta - \Delta \sigma < 0$:

$$x = p_1+\frac{p_3}{2} \uparrow  \quad \Rightarrow \quad \langle h \bar{h}\rangle \downarrow \quad \Rightarrow \quad \text{the noisy cluster} \ \Delta m \ \text{will be larger (?)} $$

so the intuition from the theoretical discussion of noise amplification is:

$$\text{we want to include less noisy neurons on second layer.}$$

Specially, condider the following two benchmarks:

First, if we have $p_1 = p_2$, we could always find $x = 0.5$, so $p_1,p_2,p_3$ will not change $\langle h \bar{h}\rangle$ (but might change $\Delta m$ here, not sure).

Second, if we have $\Delta \sigma = \Delta \eta$, $\langle h \bar{h}\rangle$ doesn't rely on $x$, or $p_1,p_2,p_3$ 

**Simulation Results**

<img src="/Users/liminhuan/Library/Application Support/typora-user-images/image-20200525234150823.png" alt="image-20200525234150823" style="zoom:50%;" />





### 3. Readout Error







