package neuralnetworksblas

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
	Verbose      bool

	Nlayers int
	Bias    []*mat.Dense
	Weights []*mat.Dense
}

func NewMultiLayerPerceptron() *MultiLayerPerceptron {
	return &MultiLayerPerceptron{
		Epochs:       100,
		BatchSize:    32,
		LearningRate: 1e-2,
		Verbose:      true,
		Activation:   "relu",
	}
}

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
}

var Derivative = map[string]func(x *mat.Dense){
	"identity": func(x *mat.Dense) {
		rows, cols := x.Dims()
		for i := range rows {
			for j := range cols {
        x.Set(i, j , 1)
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

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidDerivative(z float64) float64 {
	s := sigmoid(z)
	return s * (1 - s)
}

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

func (mlp *MultiLayerPerceptron) Train(X, y *mat.Dense) {
	mlp.initWeights()

	nSamples, features := X.Dims()
	_, ycols := y.Dims()

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

			Xs := X.Slice(batchStart, batchEnd, 0, features).(*mat.Dense)
			ys := y.Slice(batchStart, batchEnd, 0, ycols).(*mat.Dense)

			weightGrads, biasGrads := mlp.backprop(Xs, ys)

			mlp.updateParams(weightGrads, biasGrads)

		}
		activatations, _ := mlp.forwardPass(X)
		loss := mlp.loss(activatations[len(activatations)-1], y)

		if mlp.Verbose {
			fmt.Printf("Epoch %v, loss: %v, time:%v\n", i, loss, time.Since(t0))
		}

		t0 = time.Now()
	}

	fmt.Println("Training finished, time taken :", time.Since(t1))

	// make a prediction
	xPredict := X.Slice(2, 3, 0, features).(*mat.Dense)
	yPredict := y.Slice(2, 3, 0, 10).(*mat.Dense)
	fmt.Printf("yPredict: %v\n", yPredict)
	act, _ := mlp.forwardPass(xPredict)
	fmt.Printf("act: %v\n", act[len(act)-1])
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

	for i := range mlp.Nlayers - 1 {

		var z mat.Dense
		z.Mul(activations[i], mlp.Weights[i])
		activations[i+1] = &z
		addIntercepts(*activations[i+1], *mlp.Bias[i])

		var a mat.Dense
		a.CloneFrom(&z)
		zs[i] = &a
		activatezs(activations[i+1])
	}

	return activations, zs
}

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

func (mlp *MultiLayerPerceptron) backprop(X, y *mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	activations, zs := mlp.forwardPass(X)
	derivativeZ := Derivative[mlp.Activation]
	nSamples, _ := X.Dims()
	layer := mlp.Nlayers - 2

	weightGrads := make([]*mat.Dense, mlp.Nlayers-1)
	biasGrads := make([]*mat.Dense, mlp.Nlayers-1)
	deltas := make([]*mat.Dense, mlp.Nlayers-1)

	//getting the error of the output layer
	var delta mat.Dense
	delta.Sub(activations[len(activations)-1], y)
	derivativeZ(zs[layer])
	delta.MulElem(&delta, zs[layer])
	deltas[layer] = &delta

	mlp.calculateLossGrads(weightGrads, biasGrads, deltas, activations[len(activations)-2], layer, nSamples)

	//propagate that error backwards
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

func (mlp *MultiLayerPerceptron) updateParams(weightGrads, biasGrads []*mat.Dense) {
	for i := range len(weightGrads) {
		mlp.Weights[i].Sub(mlp.Weights[i], weightGrads[i])
	}

	for i := range len(biasGrads) {
		mlp.Bias[i].Sub(mlp.Bias[i], biasGrads[i])
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

func (mlp *MultiLayerPerceptron) loss(y, h *mat.Dense) float64 {
	//Mean squared error
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
}

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
