**Slide 1:**

Hi everyone. Thanks for join the discussion.

For those who don't know me before, i am currently a G1phd student in applied physics. And I am taking a research course with Cengiz this semester. 

Today, i am going to share the project i've been doing last 2 months. It mainly follows the topic of Arvin's master thesis, about the mixed selectivity in multimodal information source. We made several modifications and produce some new results, try to make the message of the whole project more clear. 

Also, this is an unfinished project for me, so your suggestions and comments for the future directions are super welcome. 

Ok, let's start 



**Slide 2:**

This is the outline of today's sharing.

First, i will give a big picture of the project, especially about the motivation, to make it more comprehensible by audience.

Then, i will give the details of the computational model, followed by some interesting results. 

After that, i will summarize the message and discuss about future direction.



**Slide 3:**

Let's ask a general question:

In our everyday life, we encouter various kinds of information, like we see something, we see its color, we see it's position, we touch it, we smell it,et.c. How could brain cooperate different kinds of stimuli and perform multiple tasks? 

Maybe it has different neurons specifed for different things. 

Or Neurons can do more than one things. 

The first one is true for lower level sensory systems.

But, let's see an experimental fact:  when monkeys are trained on a cognitive-demanding tasks, about forty percent of neurons in its Prefrontal Cortex are engaged. 

If each neurons in PFC could do only one thing, monkeys only learn 2-3 task before reaching the capacity. 

So, especially for higher level neurons, they are more likely to be able to respond to multiple stimuli. This property is termed 'mixed selectivety'. 

Then comes the next question? Why did the brain develop this property? Why not just let each neuron or brain area to do one thing? it seems easier. 



**Slides 4 - 7:**

To solve it, let's formulate this question into a more addressable framework.

We have multimodal information sources here, each modality represents one kind of stimuli. 

Then we will send those input to higher cortical layer, followed by a classic hebbian readout to do pattern seperation. 

If each neuron on cortical layers could speak to only one kind of input, it will look like these. Different parts correspond to different input.

Or they can respond to different sources like this. 

So our task is to find out, how would different strategies in the first step change the following linear separation? 

Does mixing selectivity increase the capacity? or does it make the seperation more robust under noise?  We can try to answer these more direct questions with our computational model.  



**Slides 8:**

Here are some details with the model. 

Assume we have 3 modalities now. 

$\xi^\mu\in\{-1,1\}^N, \mu = 1,\dots,P$

$\sigma^\mu\in\{-1,1\}^M, \mu = 1,\dots,K$

$\eta^\mu\in\{-1,1\}^M, \mu = 1,\dots,K$

**Slides 9:**

$h_{i}^{\xi^\mu}=\sum_{j} J_{i j}^{(\xi)} \xi_{j}^{\mu}, \quad h_{i}^{\sigma^\mu}=\sum_{j} J_{i j}^{(\sigma)} \sigma_{j}^{\mu}, \quad h_{i}^{\eta^\mu}=\sum_{k} J_{i k}^{(\eta)} \eta_{k}^{\mu}$ 

$h = (h^\xi,h^\sigma,h^\eta) = (J^\xi\xi,J^\sigma\sigma,J^\eta\eta)$

$h = (h^{\xi,\eta},h^{\xi,\sigma},h^{\sigma,\eta} )$

$h^{\xi,\eta} = J^\xi\xi+J^\eta\eta$

$h = h^{\xi,\sigma,\eta} = (J^\xi\xi+J^\sigma\sigma,J^\eta\eta)$

$J$

