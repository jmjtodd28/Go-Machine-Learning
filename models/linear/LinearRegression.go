// Simple OLS implementation of Linear Regression

package models

import (
	"Go-Machine-Learning/utils"
	"errors"
)

var (
	ERRShape        = errors.New("Incorrect shapes, linear regression requires X and y to have the same number of rows")
	ErrNotTrained   = errors.New("Cannot predict for untrained model")
	ERRPredictShape = errors.New("Shape mismatch, shape of prediction vector does not match shahpe of training data")
)

type LinearRegression struct {
	Coeffs []float64
	fitted bool
}

func NewLinearRegression() *LinearRegression{
  return &LinearRegression{}
}

func (lr *LinearRegression) Fit(X, y *utils.Matrix) error {

	if X.Rows != y.Rows {
		return ERRShape
	}

	//Create the design matrix by adding a column of 1's to the X matrix
	newCols := X.Cols + 1
	newData := make([]float64, X.Rows*newCols)

	for i := range X.Rows {
		newData[i*newCols] = 1.0
		for j := range X.Cols {
			newData[i*newCols+j+1] = X.Data[i*X.Cols+j]
		}
	}

	XDes := utils.CreateMatrix(X.Rows, newCols, newData)

	//X Transposed
	XT := *XDes
	XT.T()

	// Calculate (X^T * X)
	A := &utils.Matrix{}
	A.Dot(&XT, XDes)

	//(X^T * X)^-1
	A.Inverse()

	// calculate (X^T * y)
	B := &utils.Matrix{}
	B.Dot(&XT, y)

	//calculate the final(X^T * X)^-1 * (X^T * y)
	var r utils.Matrix
	r.Dot(A, B)

	lr.Coeffs = r.Data
	lr.fitted = true

	return nil
}

// y = w0 + w1x1 + w2x2 + ... wnxn
func (lr *LinearRegression) Predict(X *utils.Matrix) (float64, error) {

	if !lr.fitted {
		return 0.0, ErrNotTrained
	}

	if len(X.Data) != len(lr.Coeffs)-1 {
		return 0.0, ERRPredictShape
	}

	yHat := lr.Coeffs[0]

	for i, x := range X.Data {
		yHat += x * lr.Coeffs[i+1]
	}

	return yHat, nil

}

