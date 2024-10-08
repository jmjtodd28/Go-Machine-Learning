package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type MultiLayerPerceptron struct {
	Arch         []int
	Epochs       int
	BatchSize    int
	LearningRate float64
	Activation   string
	LossFunction string
	Momentum     float64
	Verbose      bool

	Nlayers int
	Bias    []*mat.Dense
	Weights []*mat.Dense

	Fitted       bool
	IsClassifier bool
	// output layer activation layer depends on whether it is a classification or regression problem
	OutputActivation string

	//used for momentum SGD
	weightVelocities []*mat.Dense
	biasVelocities   []*mat.Dense
}

func NewMultiLayerPerceptron() *MultiLayerPerceptron {
	return &MultiLayerPerceptron{
		Epochs:       100,
		BatchSize:    32,
		LearningRate: 1e-2,
		Verbose:      true,
		Activation:   "relu",
		Fitted:       false,
		Momentum:     0.9,
		IsClassifier: true,
		LossFunction: "crossEntropyLoss",
	}
}

// activation functions applied over a layer
var Activate = map[string]func(x *mat.Dense){
	"identity": func(x *mat.Dense) {},
	"sigmoid": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := sigmoid(x.At(i, j))
				x.Set(i, j, value)
			}
		}
	},
	"relu": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := x.At(i, j)
				if value < 0 {
					x.Set(i, j, 0)
				}
			}
		}
	},
	"tanh": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := math.Tanh(x.At(i, j))
				x.Set(i, j, value)
			}
		}
	},
	"softmax": func(x *mat.Dense) {
		rows, cols := x.Dims()

		for i := range rows {
			sum := 0.0
			for j := range cols {
				value := math.Exp(x.At(i, j))
				x.Set(i, j, value)
				sum += value
			}

			for j := range cols {
				x.Set(i, j, (x.At(i, j) / sum))
			}
		}
	},
}

var Derivative = map[string]func(x *mat.Dense){
	"identity": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				x.Set(i, j, 1)
			}
		}
	},
	"sigmoid": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := sigmoidDerivative(x.At(i, j))
				x.Set(i, j, value)
			}
		}
	},
	"relu": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := x.At(i, j)
				if value > 0 {
					x.Set(i, j, 1)
				} else {
					x.Set(i, j, 0)
				}
			}
		}
	},
	"tanh": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
				value := 1 - math.Pow(math.Tanh(x.At(i, j)), 2)
				x.Set(i, j, value)
			}
		}
	},
}

var LossFunctions = map[string]func(y, h *mat.Dense) float64{
	"MSELoss": func(y, h *mat.Dense) float64 {
		sum := 0.0
		var loss mat.Dense
		loss.Sub(y, h)
		rows, cols := loss.Dims()
		for i := range rows {
			for j := range cols {
				value := loss.At(i, j)
				sum += value * value
			}
		}
		return sum / (2 * float64(y.RawMatrix().Rows))
	},
	"crossEntropyLoss": func(y, h *mat.Dense) float64 {
		sum := 0.0
		rows, cols := y.Dims()
		for i := range rows {
			for j := range cols {
				if y.At(i, j) != 0 {
					sum += -math.Log(h.At(i, j))
				}
			}
		}
		return sum / float64(rows)
	},
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidDerivative(z float64) float64 {
	s := sigmoid(z)
	return s * (1 - s)
}

// Inits the weights of the biases and Weights for every neuron that follows normal distribution
// Each layer there is a matrix containing the weights and biases, excluding the input layer
func (mlp *MultiLayerPerceptron) initWeights() {
	mlp.Nlayers = len(mlp.Arch)
	mlp.Bias = make([]*mat.Dense, mlp.Nlayers-1)
	mlp.Weights = make([]*mat.Dense, mlp.Nlayers-1)

	for i := 1; i < len(mlp.Arch); i++ {
		biasData := make([]float64, mlp.Arch[i])
		for j := range biasData {
			biasData[j] = rand.NormFloat64() * 0.1
		}
		mlp.Bias[i-1] = mat.NewDense(1, mlp.Arch[i], biasData)

		weightData := make([]float64, mlp.Arch[i-1]*mlp.Arch[i])
		for j := range weightData {
			weightData[j] = rand.NormFloat64() * 0.1
		}
		mlp.Weights[i-1] = mat.NewDense(mlp.Arch[i-1], mlp.Arch[i], weightData)

	}
}

// Trains using SGD by splitting the data into batches
// XTest and yTest can be nil if there is no test data for training
func (mlp *MultiLayerPerceptron) Train(XTrain, yTrain, XTest, yTest *mat.Dense) {
	mlp.initWeights()

	//set the Activation of the output layer depending on problem type
	if mlp.IsClassifier {
		mlp.OutputActivation = "softmax"
	} else if !mlp.IsClassifier {
		mlp.OutputActivation = "identity"
	}

	testingData := true
	if XTest == nil && yTest == nil {
		testingData = false
	}

	loss := LossFunctions[mlp.LossFunction]

	nSamples, features := XTrain.Dims()
	_, ycols := yTrain.Dims()

	t0 := time.Now()
	t1 := time.Now()

	batchSize := mlp.BatchSize
	batchStart := 0

	for i := range mlp.Epochs {
		for batch := 0; batch*batchSize < nSamples; batch++ {
			batchStart = batch * batchSize
			batchEnd := batchStart + batchSize

			if batchEnd > nSamples {
				batchEnd = nSamples
			}

			Xs := XTrain.Slice(batchStart, batchEnd, 0, features).(*mat.Dense)
			ys := yTrain.Slice(batchStart, batchEnd, 0, ycols).(*mat.Dense)

			weightGrads, biasGrads := mlp.backprop(Xs, ys)

			mlp.updateParams(weightGrads, biasGrads)
		}

		//Calculating metrics, if there is no test data then we dont include a test loss or test accuracy
		//Printing information to screen for each epoch if verbose is true
		if mlp.Verbose {
			if mlp.IsClassifier && testingData {

				//training metrics
				trainActivatations, _ := mlp.forwardPass(XTrain)
				trainLoss := loss(yTrain, trainActivatations[len(trainActivatations)-1])
				trainAccuracy := mlp.Accuracy(yTrain, trainActivatations[len(trainActivatations)-1])
				fmt.Printf("Epoch %v, training loss: %.4f, training accuracy: %.4f%%, ", i, trainLoss, trainAccuracy)

				//testing metrics
				testActivations, _ := mlp.forwardPass(XTest)
				testLoss := loss(yTest, testActivations[len(testActivations)-1])
				testAccuracy := mlp.Accuracy(yTest, testActivations[len(testActivations)-1])
				fmt.Printf("testing loss: %.4f, testing accuracy: %.4f%% time:%v\n", testLoss, testAccuracy, time.Since(t0))

			} else if mlp.IsClassifier && !testingData {

				//train metrics
				trainActivatations, _ := mlp.forwardPass(XTrain)
				trainLoss := loss(yTrain, trainActivatations[len(trainActivatations)-1])
				trainAccuracy := mlp.Accuracy(yTrain, trainActivatations[len(trainActivatations)-1])
				fmt.Printf("Epoch %v, loss: %.4f, accuracy: %.4f%%, time:%v\n", i, trainLoss, trainAccuracy, time.Since(t0))

			} else if !mlp.IsClassifier && testingData {

				//train metrics
				trainActivatations, _ := mlp.forwardPass(XTrain)
				trainLoss := loss(yTrain, trainActivatations[len(trainActivatations)-1])
				fmt.Printf("Epoch %v, training loss: %.4f, ", i, trainLoss)

				//test metrics
				testActivations, _ := mlp.forwardPass(XTest)
				testLoss := loss(yTest, testActivations[len(testActivations)-1])
				fmt.Printf("test loss: %.4f time:%v\n", testLoss, time.Since(t0))

			} else if !mlp.IsClassifier && !testingData {

				//train metrics
				trainActivatations, _ := mlp.forwardPass(XTrain)
				trainLoss := loss(yTrain, trainActivatations[len(trainActivatations)-1])
				fmt.Printf("Epoch %v, loss: %.4f, time:%v\n", i, trainLoss, time.Since(t0))
			}
		}

		t0 = time.Now()
	}

	fmt.Println("Training finished, time taken :", time.Since(t1))

	mlp.Fitted = true

}

// Effectively returns the final activation layer of a foward pass
// regression = matrix with single prediction value
// Classification task = matrix containing predicted classes
func (mlp *MultiLayerPerceptron) Predict(X *mat.Dense) *mat.Dense {

	if !mlp.Fitted {
		panic("Model needs to be trained before making a prediction")
	}

	act, _ := mlp.forwardPass(X)
	return act[len(act)-1]
}

// does a broadcast addition of biases to matrix a
func addIntercepts(a, b mat.Dense) {
	rows, cols := a.Dims()
	for r := range rows {
		for c := range cols {
			value := a.At(r, c) + b.At(0, c)
			a.Set(r, c, value)
		}
	}
}

// Perfroms a foward pass on the network and computes the zs (linear combinations of weights and activations) and the activations
// Z^[l] = W^[l] • a^[l-1] + b^[l]
// A^[l] = g(Z^[l])
func (mlp *MultiLayerPerceptron) forwardPass(X *mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	activations := make([]*mat.Dense, len(mlp.Weights)+1)
	zs := make([]*mat.Dense, len(mlp.Weights))

	activations[0] = X
	activatezs := Activate[mlp.Activation]
	activateOutput := Activate[mlp.OutputActivation]

	for i := range mlp.Nlayers - 1 {

		var z mat.Dense
		z.Mul(activations[i], mlp.Weights[i])
		activations[i+1] = &z
		addIntercepts(*activations[i+1], *mlp.Bias[i])

		var a mat.Dense
		a.CloneFrom(&z)
		zs[i] = &a

		//for classifiers we apply the softmax to the final layer, regression just identity
		if (i + 1) == mlp.Nlayers-1 {
			activateOutput(activations[i+1])
		} else {
			activatezs(activations[i+1])
		}
	}

	return activations, zs
}

// calculate the loss gradient which will be used to update the paramaters for a specific layer
// Δw = η/m Σ (δ • (a^l-1)^T)
// Δb = η/m Σ δ
func (mlp *MultiLayerPerceptron) calculateLossGrads(weightGrads, biasGrads, deltas []*mat.Dense, activation *mat.Dense, layer, nSamples int) {
	var dw mat.Dense
	dw.Mul(activation.T(), deltas[layer])
	dw.Scale((mlp.LearningRate / float64(nSamples)), &dw)
	weightGrads[layer] = &dw

	var db *mat.Dense
	db = RowMean(deltas[layer])
	db.Scale((mlp.LearningRate / float64(nSamples)), db)
	biasGrads[layer] = db
}

// calculates the derivates with respect to each parameter and weight
func (mlp *MultiLayerPerceptron) backprop(X, y *mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	//obtain the activations and zs
	activations, zs := mlp.forwardPass(X)
	nSamples, _ := X.Dims()
	layer := mlp.Nlayers - 2
	derivativeZ := Derivative[mlp.Activation]

	weightGrads := make([]*mat.Dense, mlp.Nlayers-1)
	biasGrads := make([]*mat.Dense, mlp.Nlayers-1)
	deltas := make([]*mat.Dense, mlp.Nlayers-1)

	//getting the error of the output layer (L) so we can propagate backwards
	// δ^L = (a^L - y)
	var delta mat.Dense
	delta.Sub(activations[len(activations)-1], y)
	deltas[layer] = &delta

	mlp.calculateLossGrads(weightGrads, biasGrads, deltas, activations[len(activations)-2], layer, nSamples)

	//propagate that error backwards
	//δ^l = ((w^l)^T) • δ^l+1) ⊙ f'(Z^L)
	for l := mlp.Nlayers - 2; l >= 1; l-- {
		var newDelta mat.Dense
		newDelta.Mul(deltas[l], mlp.Weights[l].T())
		derivativeZ(zs[l-1])
		newDelta.MulElem(&newDelta, zs[l-1])
		deltas[l-1] = &newDelta

		mlp.calculateLossGrads(weightGrads, biasGrads, deltas, activations[l-1], l-1, nSamples)

	}
	return weightGrads, biasGrads
}

// updates all the weights and biases with momentum
// w = w - Δw
func (mlp *MultiLayerPerceptron) updateParams(weightGrads, biasGrads []*mat.Dense) {
	// on the first iteration we need to init the velocities to zero
	if mlp.weightVelocities == nil {
		mlp.weightVelocities = make([]*mat.Dense, len(weightGrads))
		mlp.biasVelocities = make([]*mat.Dense, len(biasGrads))
		for i := range mlp.weightVelocities {
			r, c := weightGrads[i].Dims()
			mlp.weightVelocities[i] = mat.NewDense(r, c, make([]float64, r*c))
		}
		for i := range mlp.biasVelocities {
			r, c := biasGrads[i].Dims()
			mlp.biasVelocities[i] = mat.NewDense(r, c, make([]float64, r*c))
		}
	}

	for i := range len(weightGrads) {
		//updating Weights
		// v = beta * v - dw
		var updateW mat.Dense
		updateW.Scale(mlp.Momentum, mlp.weightVelocities[i])
		updateW.Sub(&updateW, weightGrads[i])
		mlp.weightVelocities[i].Copy(&updateW)

		// w = w + v
		mlp.Weights[i].Add(mlp.Weights[i], mlp.weightVelocities[i])

		// Updating biases
		// v = beta * v - db
		var updateB mat.Dense
		updateB.Scale(mlp.Momentum, mlp.biasVelocities[i])
		updateB.Sub(&updateB, biasGrads[i])
		mlp.biasVelocities[i].Copy(&updateB)

		// b = b + v
		mlp.Bias[i].Add(mlp.Bias[i], mlp.biasVelocities[i])
	}
}

func RowMean(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	x := mat.NewDense(1, cols, make([]float64, cols))
	for i := range cols {
		sum := 0.0
		for j := range rows {
			sum += m.At(j, i)
		}
		x.Set(0, i, sum/float64(rows))
	}
	return x
}

// accuracy for clasification tasks to display % of predicted correct
func (mlp *MultiLayerPerceptron) Accuracy(y, h *mat.Dense) float64 {
	rows, _ := y.Dims()
	correct := 0.0
	for i := range rows {
		row := h.RawRowView(i)
		maxIdx := argmax(row)
		if y.At(i, maxIdx) == 1.0 {
			correct++
		}
	}
	return (correct / float64(rows)) * 100
}

// Function that returns index of largest value in a matrix
func argmax(x []float64) int {
	maxVal := 0.0
	maxIdx := 0
	for i := range x {
		value := x[i]
		if value > maxVal {
			maxIdx = i
			maxVal = value
		}
	}
	return maxIdx
}

// Function to print the weights and biases for each layer
func (mlp *MultiLayerPerceptron) Printnn() {
	for i := range mlp.Weights {
		fa := mat.Formatted(mlp.Weights[i], mat.Prefix(" "), mat.Squeeze())
		fmt.Println("layer ", i, " weights :\n", fa)
	}
	for i := range mlp.Bias {
		fa := mat.Formatted(mlp.Bias[i], mat.Prefix(" "), mat.Squeeze())
		fmt.Println("layer ", i, " Bias :\n", fa)
	}
}
