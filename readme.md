# Minimal Web Neural Network

This is an tiny example of a NN implmented in TypeScript, designed to be used in simple demonstrations of how machine learning works, or for just for fun. 

Either the contents of `library.ts` can be used directly in applications, or to generate and use the JavaScript in a web application, clone this repository and in a terminal;

Install the packages:
```
npm i
```

To compile the TS:
```
npx tsc
```

This will output the main library code to `./dist/matrix.js` and can be used in an application.

## Usage

The main part of this library that will be used is the `Network` class, in this example we will go through setting up a network for predicting the output of a XOR gate;

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 1       | 0       | 1      |
| 0       | 1       | 1      |
| 1       | 1       | 0      |

<sub>The XOR gate will output 1 when both of its inputs are different.</sub>

First we need to make a network, a network is made of connected layers of "neurons", each neuron accepts a number and outputs to every neuron in the next row:

<img src="https://i.ibb.co/g7kpwBd/basic-layer.png" />

The first and final layers are the input and outs of the network respectively, so a network with 2 input neuron, and two output neurons will; accept two numbers, and will try to predict what the 2 output numbers will be. 

<img src="https://i.ibb.co/zVnW030/NN.png" />

<br />

To create the network for a XOR gate, as we can see from the above table it has 2 input numbers and one output number, our library will accept an array of numbers saying how many neurons each layer will have, in this case, we will need 2 neurons on the input, and 1 neuron on the output. 

```
let network = new Network([2, 1]...
```

This is good, but we will need more than just 2 layers for our network to be accurate, so lets add a middle (or "hidden") layer that will beef up our network, we can add as many neurons as we want here.

```
let network = new Network([2, 3, 1]...
```

Now each neuron will process its input value, we need to specify in what way it should process it, and how to do the reverse of it so we can "see" how each neuron went wrong later and make adjustments. This is called an "activation" function, and we have included a fairly simple one called SIGMOID.

```
let network = new Network([2, 3, 1], SIGMOID...
```

Now when we get to making corrections in the network, we don't want to huge changes too quickly, we need to limmit the amount it changes, so we then specify our "learning rate".

```
let network = new Network([2, 3, 1], SIGMOID, 0.5);
```

And now we have out network! 

<br />

Now we just need to train it. 

There are 2 parts to training a network, the first, is to find out where we are wrong, we need to see what the network, we need to give it some inputs and see what the output is, this is called feeding forward. 

```
network.feedForwards([1, 0]);
```

We give it an array of 2 numbers, since a XOR gate has 2 inputs, and our network has 2 neurons expecting 2 inputs, feeding more or less numbers will make the network throw an error. 

Now we need to show the network what it predicted, and what it should have predicted, so that way it may go through and make tweaks to itself, this is called propagation, so we need to store the output of the feeding forward in a variable called `res` and pass it to the `backPropagate` method.

```
let res = network.feedForwards([1, 0]);
network.backPropagate(res, [1]);
```

We pass the method the expected result, since the network has 1 neuron on the output layer, we pass it a single number, and since the output was for the inputs `[1, 0]` which if we refer to the table above, should produce the output `1`.

Now if we try and see the output of calling the `feedForwards` method again, we'll see the output isn't where we'd really want it to be:

```
let res = network.feedForwards([1, 0]);
network.backPropagate(res, [1]);
res = network.feedForwards([1, 0]);
console.log(res); // Prints "[0.5205562960067879]"
```

That's because we need to actually repeat this process of forward feeding, and back propagating thousands of times with all the possible inputs and outputs. 

```
let res = network.feedForwards([0, 0]);
network.backPropagate(res, [0]);
res = network.feedForwards([1, 0]);
network.backPropagate(res, [1]);
res = network.feedForwards([0, 1]);
network.backPropagate(res, [1]);
...
```

Sounds like a lot! Right...

Thankfully we have another method on our network class that will do this for us, we need to specify an array of inputs, and an array of corresponding outputs, and the number of times we want to repeat this (called "epochs"), so lets use:

```
// Each input index corresponds to each output index 
let inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
];

let outputs = [
    [0.0],
    [1.0],
    [1.0],
    [0.0],
];

// 10,000 is a good amount of epochs
network.train(inputs, outputs, 10000); 
```

Now if we give the network an input, we get something closer to what we want.

```
console.log(network.feedForwards([0, 1]));
// Prints [0.982976055192546]
```

That's a little off, but that's ok, neural networks will never actually approach absolute numbers, instead, we can just use tools like `Math.round`

```
let result = network.feedForwards([0, 1]);
console.log(Math.round(result[0]));
// Prints 1
```

And that's it! You can play with specifying different learning rates, epochs and more hidden layers with more/less neurons - just remember the first and last must match the number of inputs and outputs, and see how all this affects the accuracy of the network. 
