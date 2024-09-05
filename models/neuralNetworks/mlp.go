package neuralnetworks

import (
	"Go-Machine-Learning/utils"
	"fmt"
	"math"
	"math/rand"
	"time"
)

type MultiLayerPerceptron struct {
	Arch         []int
	Epochs       int
	BatchSize    int
	LearningRate float64

	Bias    []*utils.Matrix
	Weights []*utils.Matrix
}

func NewMultiLayerPerceptron() *MultiLayerPerceptron {
	return &MultiLayerPerceptron{
		Epochs:       100,
		BatchSize:    32,
		LearningRate: 1e-2,
	}
}

func (mlp *MultiLayerPerceptron) InitWeights() {

	mlp.Bias = make([]*utils.Matrix, len(mlp.Arch)-1)
	mlp.Weights = make([]*utils.Matrix, len(mlp.Arch)-1)

	// start at i = 1 becuase the first input layer has no weight/bias
	for i := 1; i < len(mlp.Arch); i++ {

		biasData := make([]float64, mlp.Arch[i])
		for j := range biasData {
			biasData[j] = rand.NormFloat64()
		}
		mlp.Bias[i-1] = utils.CreateMatrix(mlp.Arch[i], 1, biasData)

		weightData := make([]float64, mlp.Arch[i-1]*mlp.Arch[i])
		for j := range weightData {
			weightData[j] = rand.NormFloat64()
		}
		mlp.Weights[i-1] = utils.CreateMatrix(mlp.Arch[i], mlp.Arch[i-1], weightData)

	}

}

// Perfroms a foward pass on the network and computes the zs (linear combinations of weights and activations) and the activations
// Z^[l] = W^[l] • a^[l-1] + b^[l]
// A^[l] = g(Z^[l])
func (mlp *MultiLayerPerceptron) ForwardPass(X *utils.Matrix) ([]*utils.Matrix, []*utils.Matrix) {
	activations := make([]*utils.Matrix, len(mlp.Weights)+1)
	zs := make([]*utils.Matrix, len(mlp.Weights))

	activations[0] = X
	for i := range mlp.Weights {
		z := utils.Dot(mlp.Weights[i], X)
		z = utils.AddB(z, mlp.Bias[i])
		zs[i] = z
		X = z.MatCopy()
		SigmoidM(X)
		activations[i+1] = X
	}
	return activations, zs
}

func (mlp *MultiLayerPerceptron) BackProp(X, y *utils.Matrix) ([]*utils.Matrix, []*utils.Matrix) {

	a := X.MatCopy()
	ys := y.MatCopy()

	activations, zs := mlp.ForwardPass(a)

	//Backwards Pass
	//derivatives fo weights and biasesa are stored here
	nabla_w := make([]*utils.Matrix, len(mlp.Weights))
	nabla_b := make([]*utils.Matrix, len(mlp.Weights))

	//output error for the final layer stored in delta
	//delta = (a^L - y) * sigma'(Z)
	var delta utils.Matrix

	//(a^L - y)
	act := activations[len(activations)-1]
	delta = *utils.Subtract(act, ys)

	// sigma'(Z) for the final layer
	sigmaPrimeZ := zs[len(zs)-1].MatCopy()
	SigmoidPrimeM(sigmaPrimeZ)

	//delta = (a^L - y) * sigma'(Z)
	delta = *utils.Multiply(&delta, sigmaPrimeZ)

	//db is bias partial derive averaged across the rows
	db := delta.MatCopy()
	meandb := RowMean(db)
	nabla_b[len(nabla_b)-1] = meandb

	//(a^(x, l-1))^T
	activations[len(activations)-2].T()
	//nabla_w = delta • (a^(x, l-1))^T
	dw := *utils.Dot(&delta, activations[len(activations)-2])
	nabla_w[len(nabla_w)-1] = &dw

	//propagate error through l layers and gather gradients
	//delta = (w^(l+1)^T • delta^(l+1)) ⊙ σ′(z^(x,l))
	for l := len(mlp.Weights) - 2; l >= 0; l-- {

		//((w^(l+1))^T
		wPlusOne := mlp.Weights[l+1].MatCopy()
		wPlusOne.T()

		//((w^(l+1))^T • delta^(l+1))
		delta = *utils.Dot(wPlusOne, &delta)

		//σ′(z^(x,l))
		sigmaPrimeZ := zs[l].MatCopy()
		SigmoidPrimeM(sigmaPrimeZ)

		//delta = (w^(l+1)^T • delta^(l+1)) ⊙ σ′(z^(x,l))
		delta = *utils.Multiply(&delta, sigmaPrimeZ)

		//grad_b = delta
		db := delta.MatCopy()
		db = RowMean(db)
		nabla_b[l] = db

		//grad_w = delta • (a^l-1)^T
		activations[l].T()
		dw := utils.Dot(&delta, activations[l])
		nabla_w[l] = dw
	}

	return nabla_b, nabla_w

}

// Weights = Weights - learningRate * dw
// bias = bias - LearningRate * db
func (mlp *MultiLayerPerceptron) UpdateParams(nabla_b, nabla_w []*utils.Matrix) {

	for i := range nabla_w {
		nabla_w[i].MultElem(mlp.LearningRate)
		mlp.Weights[i] = utils.Subtract(mlp.Weights[i], nabla_w[i])
	}

	for i := range nabla_b {
		nabla_b[i].MultElem(mlp.LearningRate)
		mlp.Bias[i] = utils.Subtract(mlp.Bias[i], nabla_b[i])
	}
}

func (mlp *MultiLayerPerceptron) Loss(y, h *utils.Matrix) float64 {
	sum := 0.0
	E := utils.Subtract(y, h)

	for i := range E.Data {
		sum += E.Data[i] * E.Data[i]
	}

	return sum / (2 * float64(len(y.Data)))
}

func (mlp *MultiLayerPerceptron) Train(X, y *utils.Matrix) {

	mlp.InitWeights()

	XT := X.MatCopy()
	XT.T()

	yT := y.MatCopy()
	yT.T()

	//Stochastic Gradiant Descent with the use of minibatch
	batchSize := mlp.BatchSize
	batchStart := 0

	t0 := time.Now()

	rows, _ := X.Dims()

	for i := range mlp.Epochs {

		//		utils.ShuffleRows(X, y)

		for batch := 0; batch*batchSize < rows; batch++ {
			batchStart = batch * batchSize
			batchEnd := batchStart + batchSize

			if batchEnd > rows {
				batchEnd = rows
			}

			Xs := X.RowSlice(batchStart, batchEnd)
			ys := y.RowSlice(batchStart, batchEnd)

			Xs.T()
			ys.T()
			nabla_b, nabla_w := mlp.BackProp(Xs, ys)

			mlp.UpdateParams(nabla_b, nabla_w)

		}
		activations, _ := mlp.ForwardPass(XT)
		loss := mlp.Loss(activations[len(activations)-1], yT)
		fmt.Printf("Epoch %v, loss: %v, time:%v\n", i, loss, time.Since(t0))
		t0 = time.Now()

	}

	fmt.Println("------------")

	x := X.Row(2).MatCopy()
	x.T()
	fmt.Println("predict:")
	y.Row(2).PrintMatrix()
	act, _ := mlp.ForwardPass(x)

	act[len(act)-1].PrintMatrix()


}

func (mlp *MultiLayerPerceptron) Printnn() {

	for i := range len(mlp.Weights) {
		if i+1 != len(mlp.Weights) {
			fmt.Println("Hidden Layer", i+1)
		} else {
			fmt.Println("Ouptut Layer (layer", i+1, ")")
		}
		fmt.Println("Weights")
		mlp.Weights[i].PrintMatrix()
		fmt.Println("bias")
		mlp.Bias[i].PrintMatrix()
	}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// applies the sigmoid to the whole matrix
func SigmoidM(m *utils.Matrix) {
	for i := range m.Data {
		m.Data[i] = Sigmoid(m.Data[i])
	}
}

func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}

func SigmoidPrimeM(m *utils.Matrix) {
	for i := range m.Data {
		m.Data[i] = SigmoidPrime(m.Data[i])
	}
}

// find the mean of each row
// a 4x4 would become a 4x1 where each row is average of the 4 points
func RowMean(m *utils.Matrix) *utils.Matrix {
	rows, cols := m.Dims()
	x := &utils.Matrix{
		Rows: rows,
		Cols: 1,
		Data: make([]float64, rows),
	}

	for i := range rows {
		for j := range cols {
			x.Data[i] += m.Data[i*j+j]
		}
	}

	for i := range x.Data {
		x.Data[i] /= float64(cols)
	}

	return x

}

func (mlp *MultiLayerPerceptron) Predict() {
}
