package preprocessing

import (
	"log"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type StandardScaler struct {
	Mean, Std *mat.Dense
}

func NewStandardScaler() *StandardScaler {
	return &StandardScaler{}
}

// calculate the means and stds for the data
func (scaler *StandardScaler) Fit(x *mat.Dense) {

	//Init the mean and std matricies
	_, cols := x.Dims()
	scaler.Mean = mat.NewDense(1, cols, make([]float64, cols))
	scaler.Std = mat.NewDense(1, cols, make([]float64, cols))

	// go down each column and compute the mean for each column
	for i := range cols {
		var col []float64
		mean, std := stat.MeanStdDev(mat.Col(col, i, x), nil)

		scaler.Mean.Set(0, i, mean)
		scaler.Std.Set(0, i, std)
	}
}

// Transforms the data z = (x - u) / s, where u is the mean and s is the std
func (scalar *StandardScaler) Transform(x *mat.Dense) {

	rows, cols := x.Dims()
	//x needs to have the same number of columns as the mean/std matrices
	_, meanCols := scalar.Mean.Dims()
	if cols != meanCols {
		log.Fatalf("number of columns doesnt match the fitted data, Have: %d, Need: %d", cols, meanCols)
	}

	for i := range rows {
		for j := range cols {
			value := (x.At(i, j) - scalar.Mean.At(0, j)) / (scalar.Std.At(0, j))
			x.Set(i, j, value)
		}
	}
}

// does both the fit and transform at the same time
func (scalar *StandardScaler) FitTransform(x *mat.Dense) {
	scalar.Fit(x)
	scalar.Transform(x)
}
