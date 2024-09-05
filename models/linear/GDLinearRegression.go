//Linear Regression via Gradient descent

package models

import (
	"Go-Machine-Learning/utils"
	"errors"
	"fmt"
	"math"
	"time"
)

type GDLinearRegression struct {

	//parameters to be learnt
	Coeffs *utils.Matrix
	Bias   float64

	//bool to determine whether a model has been fitted before making predicitons
	Fitted bool

	//hyperparameters that are set before training
	MaxIter      int
	LearningRate float64

	//If true will print information to the console
	Verbose bool

	//Regularisation, this can be l1, l2 or none
	Regularisation string
	// Multiplies the Regularisation term
	Alpha float64

	//early stopping conditions
	// you can enable or disable early stopping
	earlyStopping bool
	//training will stop when loss > best_loss - Tol for nIterNoChange
	Tol           float64
	nIterNoChange int

	//Gradient descent type, can be batch, miniBatch (controlled by batchSize) or SGD (stochastic gradient descent)
	GDescentType string
	batchSize    int
}

// Setting some default values
func NewGDLinearRegression() *GDLinearRegression {
	return &GDLinearRegression{
		MaxIter:        1000,
		LearningRate:   1e-3,
		Fitted:         false,
		GDescentType:   "SGD",
		earlyStopping:  true,
		Tol:            1e-3,
		nIterNoChange:  5,
		Regularisation: "l2",
		batchSize:      32,
		Alpha:          1e-4,
	}
}

// Mean Squared Error
func MSE(p *utils.Matrix, y *utils.Matrix) float64 {

	var residual utils.Matrix
	residual.Subtract(p, y)
	residualT := residual
	residualT.T()

	var MSE utils.Matrix
	MSE.Dot(&residualT, &residual)

	if len(MSE.Data) > 1.0 {
		panic("MSE has length longer than 1 value:")
	}

	MSEVal := MSE.Data[0] / float64(2.0*y.Rows)

	return MSEVal
}

// Makes new predictions based on updated weights
func NewPredictions(X *utils.Matrix, glr *GDLinearRegression) *utils.Matrix {
	var p utils.Matrix
	p.Dot(X, glr.Coeffs)
	p.AddElem(glr.Bias)
	return &p
}

// Uses Mean squared error
func (glr *GDLinearRegression) calculateBatchGradients(X, y, p *utils.Matrix) (float64, *utils.Matrix) {

	// XT is the transposition of batch X
	XT := X.MatCopy()
	XT.T()

	var residual utils.Matrix
	residual.Subtract(p, y)

	N := 2.0 / float64(y.Rows)
	BiasGrad := residual.Sum() * N

	var gradients utils.Matrix
	gradients.Dot(XT, &residual)
	gradients.MultElem(N)

	//Add Regularisation
	if glr.Regularisation == "l2" {
		var regularisation utils.Matrix
		var regularisedGradients utils.Matrix

		regularisation = *glr.Coeffs.MatCopy()
		regularisation.MultElem(glr.Alpha)
		regularisedGradients.Add(&gradients, &regularisation)

		return BiasGrad, &regularisedGradients
	} else if glr.Regularisation == "l1" {
		for i := range gradients.Data {
			gradients.Data[i] += glr.Alpha * sign(gradients.Data[i])
		}

		return BiasGrad, &gradients
	} else if glr.Regularisation == "none" {
		return BiasGrad, &gradients
	} else {
		panic("glr.regularisation must be \"l1\", \"l2\", or \"none\"")
	}
}

// returns 1 x > 0, 0 if x = 0 and -1 if x < 0
func sign(x float64) float64 {
	if x > 0 {
		return 1.0
	} else if x < 0 {
		return -1.0
	} else {
		return 0.0
	}
}

// w = w - eta * gradients
func (glr *GDLinearRegression) UpdateCoefficients(gradients *utils.Matrix) {
	gradients.MultElem(glr.LearningRate)

	var newCoeffs utils.Matrix
	newCoeffs.Subtract(glr.Coeffs, gradients)

	glr.Coeffs = &newCoeffs

}

// b = b - eta * bias gradiant
func (glr *GDLinearRegression) UpdateBias(gradient float64) {
	glr.Bias = glr.Bias - glr.LearningRate*gradient
}

// Fits the coefficients and the bias using gradient descent
func (glr *GDLinearRegression) Fit(X *utils.Matrix, y *utils.Matrix) error {

	if y.Cols != 1 {
		return errors.New("output data must have one column of data")
	}

	if X.Rows != y.Rows {
		return errors.New("number of examples need to match between input and output data")
	}

	//Learning rate cannot be less than or equal to 0
	if glr.LearningRate <= 0 {
		return errors.New("Learning rate cannot be less than or equal to zero")
	}

	// Init the coefficients and Bias to zero
	glr.Coeffs = &utils.Matrix{Rows: X.Cols, Cols: 1, Data: make([]float64, X.Cols)}

	//Init the gradients of the Bias and coefficients
	BiasGradient := 0.0
	var gradients *utils.Matrix

	//start the timer
	t0 := time.Now()

	//early stopping values
	bestLoss := math.Inf(1)
	iterNoImprov := 0

	if glr.GDescentType == "batch" {

		for i := range glr.MaxIter {

			t1 := time.Now()

			p := *NewPredictions(X, glr)
			BiasGradient, gradients = glr.calculateBatchGradients(X, y, &p)

			glr.UpdateBias(BiasGradient)
			glr.UpdateCoefficients(gradients)

			MSE := MSE(&p, y)

			if glr.Verbose {
				fmt.Printf("-- Iteration %d\nBias: %v, Error: %v, Epoch Time: %v\n", (i + 1), glr.Bias, MSE, time.Since(t1))
			}

			//check for early stopping
			if glr.earlyStopping {
				if MSE < bestLoss-glr.Tol {
					bestLoss = MSE
					iterNoImprov = 0
				} else if MSE > bestLoss-glr.Tol {
					iterNoImprov++
				}

				if iterNoImprov == glr.nIterNoChange {
					if glr.Verbose {
						fmt.Println("Convergence after", i, "epochs took", time.Since(t0))
					}
					break
				}
			}

		}
	} else if glr.GDescentType == "SGD" {

		for j := range glr.MaxIter {
			t1 := time.Now()
			utils.ShuffleRows(X, y)
			for i := range X.Rows {

				x_sample := X.Row(i)
				y_sample := y.Row(i)

				p := *NewPredictions(x_sample, glr)
				BiasGradient, gradients = glr.calculateBatchGradients(x_sample, y_sample, &p)

				glr.UpdateBias(BiasGradient)
				glr.UpdateCoefficients(gradients)

			}

			p := *NewPredictions(X, glr)
			MSE := MSE(&p, y)

			if glr.Verbose {
				fmt.Printf("-- Iteration %d\nBias: %v, Error: %v, Epoch Time: %v\n", (j + 1), glr.Bias, MSE, time.Since(t1))
			}
			//check for early stopping
			if glr.earlyStopping {
				if MSE < bestLoss-glr.Tol {
					bestLoss = MSE
					iterNoImprov = 0
				} else if MSE > bestLoss-glr.Tol {
					iterNoImprov++
				}

				if iterNoImprov == glr.nIterNoChange {
					if glr.Verbose {
						fmt.Println("Convergence after", j, "epochs took", time.Since(t0))
					}
					break
				}
			}
		}

	} else if glr.GDescentType == "miniBatch" {

		miniBatchSize := glr.batchSize
		miniBatchStart := 0

		for j := range glr.MaxIter {
			t1 := time.Now()
			utils.ShuffleRows(X, y)

			for miniBatch := 0; miniBatch*miniBatchSize < X.Rows; miniBatch++ {
				miniBatchStart = miniBatch * miniBatchSize
				miniBatchEnd := miniBatchStart + miniBatchSize

				if miniBatchEnd > X.Rows {
					miniBatchEnd = X.Rows
				}

				Xs := X.RowSlice(miniBatchStart, miniBatchEnd)
				ys := y.RowSlice(miniBatchStart, miniBatchEnd)

				p := *NewPredictions(Xs, glr)

				BiasGradient, gradients := glr.calculateBatchGradients(Xs, ys, &p)

				glr.UpdateBias(BiasGradient)
				glr.UpdateCoefficients(gradients)

			}

			p := *NewPredictions(X, glr)
			MSE := MSE(&p, y)

			if glr.Verbose {
				fmt.Printf("-- Iteration %d\nBias: %v, Error: %v, Epoch Time: %v\n", (j + 1), glr.Bias, MSE, time.Since(t1))
			}
			//check for early stopping
			if glr.earlyStopping {
				if MSE < bestLoss-glr.Tol {
					bestLoss = MSE
					iterNoImprov = 0
				} else if MSE > bestLoss-glr.Tol {
					iterNoImprov++
				}

				if iterNoImprov == glr.nIterNoChange {
					if glr.Verbose {
						fmt.Println("Convergence after", j, "epochs took", time.Since(t0))
					}
					break
				}
			}

		}

	} else {
		fmt.Println("The gradient descent type must be \"batch\", \"miniBatch\" or \"SGD\"")
	}

	glr.Fitted = true

	return nil
}

func (glr *GDLinearRegression) Predict(X *utils.Matrix) (float64, error) {

	if !glr.Fitted {
		return 0.0, errors.New("Need to train the model before making a prediction")
	}

	if X.Rows != 1 {
		return 0.0, errors.New("can only predict for one example at a time")
	}

	if X.Cols != len(glr.Coeffs.Data) {
		return 0.0, errors.New("shape of prediction vector does not match shape of training matrix")
	}

	var dot utils.Matrix
	dot.Dot(X, glr.Coeffs)
	yHat := glr.Bias + dot.Data[0]

	return yHat, nil
}
