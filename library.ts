
class Activation {

	constructor(func: (v: number) => number, derivative: (v: number) => number) {
		this.function = func;
		this.derivative = derivative;
	}

	function: (v: number) => number;
	derivative: (v: number) => number;
}

const SIGMOID: Activation = new Activation(v => 1.0 / (1.0 + Math.E ** -v), v => v * (1.0 - v));

class Matrix {

	constructor(rows: number, cols: number, srcOrRandom: "R" | number[][] | undefined = undefined) {
		this.rows = rows;
		this.cols = cols;
		this.data = typeof srcOrRandom === "object" ? 
			srcOrRandom : new Array(rows).fill(new Array(cols).fill(0));

		if (srcOrRandom === "R")
			this.data = this.data.map(r => r.map(_ => Math.random() * 2.0 - 1.0));
	}

	rows: number;
	cols: number;
	data: number[][];

	multiply(other: Matrix): Matrix {
		if (this.cols != other.rows) {
			throw new Error(`Tried to multiply a matrix of ${this.cols} columns by a matrix of ${other.rows} rows. `);
		}

		const data = new Array(this.rows).fill(new Array(other.cols).fill(0))
			.map((r, ri) => 
				r.map((c, ci) => 
					this.data[ri].map((tc, tci) => tc * other.data[tci][ci])
						.reduce((p, v) => p + v, 0))
			);

		return new Matrix(this.rows, other.cols, data);
	}

	add(other: Matrix): Matrix {
		if (this.cols != other.cols || this.rows != other.rows) {
			throw new Error(`Attempted to add a matrix of ${this.cols}/${this.rows} cols/rows, to a matrix of ${other.cols}/${other.rows} cols/rows. `);
		}

		const data = this.data.map((r, ri) => r.map((c, ci) => c + other.data[ri][ci]));

		return new Matrix(this.rows, other.cols, data);
	}

	dotMultiply(other: Matrix): Matrix {
		if (this.cols != other.cols || this.rows != other.rows) {
			throw new Error(`Attempted to dot multiply a matrix of ${this.cols}/${this.rows} cols/rows, by a matrix of ${other.cols}/${other.rows} cols/rows. `);
		}

		const data = this.data.map((r, ri) => r.map((c, ci) => c * other.data[ri][ci]));

		return new Matrix(this.rows, other.cols, data);
	}

	subtract(other: Matrix): Matrix {
		if (this.cols != other.cols || this.rows != other.rows) {
			throw new Error(`Attempted to subtract a matrix of ${other.cols}/${other.rows} cols/rows, from a matrix of ${this.cols}/${this.rows} cols/rows. `);
		}

		const data = this.data.map((r, ri) => r.map((c, ci) => c - other.data[ri][ci]));

		return new Matrix(this.rows, other.cols, data);
	}

	map(func: (v: number) => number): Matrix {
		const data = this.data.map(r => r.map(c => func(c)));

		return new Matrix(this.rows, this.cols, data);
	}

	transpose(): Matrix {
		let res = new Matrix(this.cols, this.rows);

		res.data = res.data.map((_, i) => _.map((_, j) => this.data[j][i]));

		return res;
	}

	clone(): Matrix {
		let res = new Matrix(this.rows, this.cols);

		res.data = this.data.map(r => r.map(c => c));

		return res;
	}
}

class Network {

	constructor(layers: number[], activation: Activation, learningRate: number) {
		this.layers = layers;
		this.weights = layers.slice(0, -1).map((l, li) => new Matrix(layers[li + 1], l, "R"));
		this.biases = layers.slice(0, -1).map((l, li) => new Matrix(layers[li + 1], 1, "R"));
		this.data = [];
		this.activation = activation;
		this.learningRate = learningRate;
	}

	layers: number[];
	weights: Matrix[];
	biases: Matrix[];
	data: Matrix[];
	activation: Activation;
	learningRate: number;

	feedForwards(inputs: number[]): number[] {
		if (inputs.length != this.layers[0]) {
			throw new Error(`Tried to feed forward inputs of size ${inputs.length}, into a network with a first layer size of ${this.layers[0]} . `);
		}

		let current = new Matrix(1, inputs.length, [inputs]).transpose();
		this.data = [current.clone()];

		for(var i = 0; i < this.layers.length - 1; i++) {
			current = this.weights[i].multiply(current.clone())
				.add(this.biases[i])
				.map(this.activation.function);
			this.data.push(current.clone());
		}

		return current.data[0];
	}

	backPropagate(outputs: number[], targets: number[]) {
		if (targets.length != this.layers[this.layers.length - 1]) {
			throw new Error(`Back propagation given ${targets.length} targets, when the final layer has a size of ${this.layers[this.layers.length - 1]}. `);
		}

		let parsed = new Matrix(1, outputs.length, [outputs]);
		let errors = new Matrix(1, targets.length, [targets]).subtract(parsed);
		let gradients = parsed.map(this.activation.derivative);

		for (var i = this.layers.length - 2; i >= 0; i--) {
			gradients = gradients.dotMultiply(errors).map(v => v * this.learningRate);

			this.weights[i] = this.weights[i].add(gradients.multiply(this.data[i].transpose()));
			this.biases[i] = this.biases[i].add(gradients);

			errors = this.weights[i].transpose().multiply(errors);
			gradients = this.data[i].map(this.activation.derivative);
		}
	}

	train(inputs: number[][], targets: number[][], epochs: number, logProgress: boolean = false) {
		for (let i = 0; i < epochs; i++) {
			if (logProgress && (epochs < 100 || i % (epochs / 100) === 0)) {
				console.log(`Epoch ${i + 1} of ${epochs}`);
			}

			for (let ii = 0; ii < inputs.length; ii++) {
				let outputs = this.feedForwards(inputs[ii]);
				this.backPropagate(outputs, targets[ii])
			}
		}
	}
}

function test() {

	let inputs = [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	];

	let targets = [
		[0.0],
		[1.0],
		[1.0],
		[0.0],
	];

	let network = new Network([2, 3, 1], SIGMOID, 0.55);

	network.train(inputs, targets, 10000);

	console.log(`0 and 0: ${network.feedForwards([0.0, 0.0])}`);
	console.log(`0 and 1: ${network.feedForwards([0.0, 1.0])}`);
	console.log(`1 and 0: ${network.feedForwards([1.0, 0.0])}`);
	console.log(`1 and 1: ${network.feedForwards([1.0, 1.0])}`);
}
