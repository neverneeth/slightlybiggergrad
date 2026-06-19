### Introduction

At the end of the day, a **neural network** is just a fancy function that takes in some input, performs a series of mathematical operations on it, and produces some output. Now to design such a function by hand, we would need to have perfect knowledge of the system the neural network attempts to model. However, in most real-world scenarios, we don't have this perfect knowledge. This is where the concept of **learning** comes into play. Instead of hand-designing the function, we can let the neural network learn it from data. The process of learning involves adjusting the parameters of the neural network (namely weights and biases) in such a way that the output of the neural network gets closer to the desired output for a given input.

Again, we are posed with a question: how do we adjust the parameters of the neural network? We could manually tune each and every parameter, until our function works for any input we might throw at it. And this could feasibly work if the number of parameters were really really small. But in practice, neural networks are employed to model highly complex systems, and thus they often have millions or even billions of parameters.

Now I'm not here to tell that it's impossible. If that's what suits you then go for it. But for other normal people(as normal as normal gets), we need something more efficient. Something that can be **automated**. This is where backpropagation and automatic differentiation come into play.

Let's assume that we have a tiny neural network as shown below. For now we assume that each neuron is associated with just a single number that we multiply with whatever input comes at us. 

FORWARD PASS  ───────────────────────────────────────────────────────────────────────────────>

   [ Input: x ]               [ Neuron 1 ]               [ Neuron 2 ]               [ Prediction ]
      (2.0)                    Weight: w1                 Weight: w2                    (ŷ)
        │                        (3.0)                      (1.5)                        │
        │                          │                          │                          │
        ▼                          ▼                          ▼                          ▼
     (  x  ) ─────────────────> [  * ] ─────────────────> [  * ] ──────────────────> (  ŷ  ) (9.0)
                                   │                          │                          │
                                   ▼                          ▼                          │   [ Target: y ]
                             [ Hidden: a ]              [ Pred: ŷ ]                      │      (10.0)
                                 (6.0)                      (9.0)                        ▼        │
                                                                                   [ Loss Node ] ◄┘
                                                                                 L = (ŷ - y)² = 1.0


<───────────────────────────────────────────────────────────────────────────────  BACKWARD PASS

  ∂L/∂w1 = -6.0              ∂L/∂a = -3.0               ∂L/∂w2 = -12.0             ∂L/∂ŷ = -2.0
  [ self.grad ]              [ local grad ]             [ param.grad ]             [ out.grad ]
       ▲                          ▲                          ▲                          ▲
       │   Multiply by x (2.0)    │   Multiply by w2 (1.5)   │   Multiply by a (6.0)    │
       └──────────────────────────┴──────────────────────────┴──────────────────────────┘

### The Tiny Linear Cascade

Let’s lay out a simple, 2-layer chain of operations. Imagine we feed an input $x$ into our first neuron, which multiplies it by a weight $w_1$. The result of that multiplication is passed to a second neuron, which multiplies it by a second weight $w_2$ to give us our final prediction, $\hat{y}$.

Mathematically, this cascade looks like a simple sequence of steps:

1. $a = x \cdot w_1$  *(Output of the first neuron)*
2. $\hat{y} = a \cdot w_2$  *(Our model's final prediction)*

To make things concrete, let's throw some real numbers into this machine. Suppose our input $x = 2.0$, and the true target value we want our network to predict is $y = 10.0$. Right now, our weights are just guessed randomly: let's say $w_1 = 3.0$ and $w_2 = 1.5$.

---

### Step 1: The Forward Pass

First, we pass our input forward through the equations to see how terribly wrong our random guesses are:

$$a = 2.0 \cdot 3.0 = 6.0$$

$$\hat{y} = 6.0 \cdot 1.5 = 9.0$$

Our network predicted **9.0**, but the real-world target is **10.0**. We missed by a full point.

To quantify our failure mathematically, we use a **Loss Function**. For regression tasks, the standard choice is Squared Error ($L$):

$$L = (\hat{y} - y)^2$$

$$L = (9.0 - 10.0)^2 = (-1.0)^2 = 1.0$$

Our loss is **1.0**. Our ultimate goal is to get this loss as close to zero as humanly possible. To do that, we need to know how changing $w_1$ and $w_2$ will affect $L$. In other words, we need to calculate the gradients: $\frac{\partial L}{\partial w_1}$ and $\frac{\partial L}{\partial w_2}$.

---

### Step 2: The Backward Pass (Unveiling the Chain Rule)

To calculate how the loss changes with respect to our weights, we have to trace our steps backward through our computational pipeline. This is where the **Chain Rule of Calculus** shines. It tells us that to find how a change at the very beginning affects the very end, we just multiply the local rates of change along the path.

Let's work backward from the Loss to the inputs.

#### 1. The Loss Gradient

How does the loss change with respect to our prediction $\hat{y}$?
Using power rule calculus on $L = (\hat{y} - y)^2$:

$$\frac{\partial L}{\partial \hat{y}} = 2 \cdot (\hat{y} - y)$$

$$\frac{\partial L}{\partial \hat{y}} = 2 \cdot (9.0 - 10.0) = -2.0$$

This tells us that if we increase our prediction $\hat{y}$ by a tiny amount, the loss will *decrease* at a rate of 2.

#### 2. Tuning the Second Neuron ($w_2$)

Next, how does the loss change with respect to our second weight, $w_2$? According to the chain rule:

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2}$$

We already know $\frac{\partial L}{\partial \hat{y}} = -2.0$. The local derivative $\frac{\partial \hat{y}}{\partial w_2}$ is just the derivative of $(a \cdot w_2)$ with respect to $w_2$, which leaves us with $a$. Therefore:

$$\frac{\partial L}{\partial w_2} = -2.0 \cdot a = -2.0 \cdot 6.0 = -12.0$$

We have our first answer! $\frac{\partial L}{\partial w_2} = -12.0$. If we increase $w_2$, the loss goes down quickly.

#### 3. Tuning the First Neuron ($w_1$)

Now, how does a change in the first weight $w_1$ down at the beginning of the network alter the final loss? We chain the derivatives all the way back:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a} \cdot \frac{\partial a}{\partial w_1}$$

Let's collect our local components:

* $\frac{\partial L}{\partial \hat{y}} = -2.0$
* $\frac{\partial \hat{y}}{\partial a} = w_2 = 1.5$  *(From $\hat{y} = a \cdot w_2$)*
* $\frac{\partial a}{\partial w_1} = x = 2.0$  *(From $a = x \cdot w_1$)*

Now we multiply them together:

$$\frac{\partial L}{\partial w_1} = -2.0 \cdot 1.5 \cdot 2.0 = -6.0$$

We have our second answer: $\frac{\partial L}{\partial w_1} = -6.0$.

---

### Step 3: The Update Step

Now that we have automated the math to give us our directional compass, we can use a basic optimization step (Gradient Descent) to adjust our parameters. We move opposite to the gradient to minimize the loss, scaled by a small learning rate ($lr = 0.01$):

$$w_1 \leftarrow w_1 - (lr \cdot \frac{\partial L}{\partial w_1}) = 3.0 - (0.01 \cdot -6.0) = 3.06$$

$$w_2 \leftarrow w_2 - (lr \cdot \frac{\partial L}{\partial w_2}) = 1.5 - (0.01 \cdot -12.0) = 1.62$$

If we run our forward pass one more time with these updated, optimized weights:


$$a = 2.0 \cdot 3.06 = 6.12$$

$$\hat{y} = 6.12 \cdot 1.62 = 9.9144$$

Look at that! Our prediction jumped from **9.0** to **9.9144**, driving our loss down from **1.0** to **0.0073**. The network is learning.

---

### Enter the Automator: Why We Build Engines

Doing this calculus by hand for two scalar parameters took an entire page of algebra. If your network has 512 hidden layers processing matrices across millions of parameters, tracking these derivatives by hand becomes physically impossible.

This is exactly why we build an **Automatic Differentiation Engine**. We don't want to calculate derivatives; we want to write code that *remembers* how it was built.

By wrapping our numbers inside a custom object that automatically saves parent relationships and execution callbacks (`_backward()`) during the forward pass, we can build a system where calling a single method—`loss.backward()`—triggers a domino effect that sweeps backward through the entire computational architecture, calculating a million complex chain-rule equations automatically.