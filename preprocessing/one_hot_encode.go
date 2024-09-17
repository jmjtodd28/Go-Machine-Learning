package preprocessing

import (
	"Go-Machine-Learning/utils"

	"gonum.org/v1/gonum/mat"
)

// n is the number of classes to one hot encode
func OneHotEncode(m *utils.Matrix, n int) {
	newMatrix := utils.CreateEmptyMatrix(m.Rows, n)
	for i, x := range m.Data {
		newMatrix.Data[i*n+int(x)] = 1.0
	}

	m.Cols = newMatrix.Cols
	m.Rows = newMatrix.Rows
	m.Data = newMatrix.Data
}

// One hot encodes the gonum dense matrix
func OneHotEncodeDense(n int, m *mat.Dense) *mat.Dense {
	rows, _ := m.Dims()

	newData := make([]float64, rows*n)

	for i, x := range m.RawMatrix().Data {
		newData[i*n+int(x)] = 1.0
	}

	newDense := mat.NewDense(rows, n, newData)

	return newDense
}
