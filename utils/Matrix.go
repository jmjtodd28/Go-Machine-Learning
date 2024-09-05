package utils

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"sync"
)

type Matrix struct {
	Rows, Cols int
	Data       []float64
}

func CreateMatrix(r, c int, data []float64) *Matrix {

	if r == 0 || c == 0 {
		panic("Cannot have a Matirx dimension = 0")
	}

	if r < 0 || c < 0 {
		panic("Cannot have negative dimensions")
	}

	if r*c != len(data) {
		panic("Dimensions do not match the data")
	}

	return &Matrix{Rows: r, Cols: c, Data: data}
}

// Creates a r x c matrix of zeros
func CreateEmptyMatrix(r, c int) Matrix {
	var emptyMat Matrix
	emptyMat.Data = make([]float64, r*c)
	emptyMat.Rows = r
	emptyMat.Cols = c

	return emptyMat
}

func (m *Matrix) At(r, c int) float64 {
	return m.Data[(r*m.Cols)+c]
}

func (m *Matrix) Dims() (int, int) {
	return m.Rows, m.Cols
}

// Element-wise addition of x and y with the data placed in the receiver m
func (m *Matrix) Add(x, y *Matrix) {

	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot add matricies with different shapes")
	}

	m.Data = make([]float64, len(x.Data))

	for i := range len(x.Data) {
		m.Data[i] = x.Data[i] + y.Data[i]
	}

	m.Rows = x.Rows
	m.Cols = x.Cols
}

// Element-wise addition of x and y and returns the result
// Broadcasting is included, two matricies can be added if they have equal dimensions or one of them is 1
func AddB(x, y *Matrix) *Matrix {

	xRow, xCol := x.Dims()
	yRow, yCol := y.Dims()

	// checking to see if dimensions are equal for matrix addition eg 3x3 and 3x3 will pass
	//	dimsEqual := xRow == yRow && xCol == yCol

	//checking to see if a dimension is 1 so we can do broadcasting matrix addition
	// eg can add a 3x3 with a 3x1 or a 1x3, but not something like a 2x3
	rowCompatible := (xRow == yRow) || (xRow == 1) || (yRow == 1)
	colCompatible := (xCol == yCol) || (xCol == 1) || (yCol == 1)
	broadcastable := rowCompatible && colCompatible

	if !broadcastable {
		panic("Cannot add matricies with different shapes or cannot broadcast together")
	}

	// Get the dimensions of the matrix that will be returned
	rows := max(xRow, yRow)
	cols := max(xCol, yCol)

	m := Matrix{
		Data: make([]float64, rows*cols),
		Rows: rows,
		Cols: cols,
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			xVal := x.Data[(i%max(1, xRow))*xCol+(j%max(1, xCol))]
			yVal := y.Data[(i%max(1, yRow))*yCol+(j%max(1, yCol))]
			m.Data[i*cols+j] = xVal + yVal
		}
	}

	return &m
}

// Element-wise substraction of x and y with the data placed in the receiver m
func (m *Matrix) Subtract(x, y *Matrix) {
	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot subtract matricies with different shapes")
	}

	m.Data = make([]float64, len(x.Data))

	for i := range x.Data {
		m.Data[i] = x.Data[i] - y.Data[i]
	}

	m.Rows = x.Rows
	m.Cols = x.Cols
}

// Element-wise substraction of x and y with the data placed in the receiver m
func Subtract(x, y *Matrix) *Matrix {

	var m Matrix
	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot subtract matricies with different shapes")
	}

	m.Data = make([]float64, len(x.Data))

	for i := range x.Data {
		m.Data[i] = x.Data[i] - y.Data[i]
	}

	m.Rows = x.Rows
	m.Cols = x.Cols

	return &m
}

// Element-wise mutliplication of x and y with the data place in the receiver m
func (m *Matrix) Multiply(x, y *Matrix) {
	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot multiply matricies with different shapes")
	}

	m.Data = make([]float64, len(x.Data))

	for i := range x.Data {
		m.Data[i] = x.Data[i] * y.Data[i]
	}

	m.Rows = x.Rows
	m.Cols = x.Cols
}

// Element-wise mutliplication of x and y with the data place in the receiver m
func Multiply(x, y *Matrix) *Matrix {
	var m Matrix
	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot multiply matricies with different shapes")
	}

	m.Data = make([]float64, len(x.Data))

	for i := range x.Data {
		m.Data[i] = x.Data[i] * y.Data[i]
	}

	m.Rows = x.Rows
	m.Cols = x.Cols

	return &m
}

// copies contents of x into receiver m such that modifying one wont affect the other
func (m *Matrix) MatCopy() *Matrix {

	copyMatrix := &Matrix{
		Rows: m.Rows,
		Cols: m.Cols,
		Data: make([]float64, len(m.Data)),
	}

	copy(copyMatrix.Data, m.Data)

	return copyMatrix
}

// Returns the nth column
func (m *Matrix) Col(n int) *Matrix {

	if n >= m.Cols {
		panic("Column index out of range")
	}

	if n < 0 {
		panic("Cannot have negative column index")
	}

	var col []float64

	for i := range m.Rows {
		col = append(col, m.Data[n+(m.Cols*i)])
	}

	return &Matrix{Rows: m.Rows, Cols: 1, Data: col}

}

// Returns the nth Row as a new Matrix.
func (m *Matrix) Row(n int) *Matrix {
	if n >= m.Rows {
		panic("Row index out of range")
	}

	if n < 0 {
		panic("Cannot have negative row index")
	}

	startIdx := n * m.Cols
	row := make([]float64, m.Cols)
	copy(row, m.Data[startIdx:startIdx+m.Cols])

	return &Matrix{Rows: 1, Cols: m.Cols, Data: row}
}

// row slice returns the rows of a sub matrix
func (m *Matrix) RowSlice(start, end int) *Matrix {
	return &Matrix{Rows: end - start, Cols: m.Cols, Data: m.Data[start*m.Cols : (end)*m.Cols]}
}

func (m *Matrix) ColSlice(start, end int) *Matrix {

	newCols := end - start
	newData := make([]float64, m.Rows*newCols)

	for r := 0; r < m.Rows; r++ {
		copy(newData[r*newCols:(r+1)*newCols], m.Data[r*m.Cols+start:r*m.Cols+end])
	}

	return &Matrix{
		Rows: m.Rows,
		Cols: newCols,
		Data: newData,
	}
}

// Dot product between two verticies
/*func Dot(x, y []float64) float64 {

	if len(x) != len(y) {
		panic("Lengths need to be the same to perform dot product")
	}

	total := 0.0
	for i := range x {
		total += x[i] * y[i]
	}
	return total
}
*/

// Dot product using parallelisation
func Dot(a, b *Matrix) *Matrix {

	if a.Cols != b.Rows {
		panic("Incorrect matrix shapes for multiplication")
	}

	var result Matrix
	result.Rows = a.Rows
	result.Cols = b.Cols
	result.Data = make([]float64, a.Rows*b.Cols)
	var wg sync.WaitGroup

	for i := 0; i < a.Rows; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			for j := 0; j < b.Cols; j++ {
				sum := 0.0
				for k := 0; k < a.Cols; k++ {
					sum += a.Data[row*a.Cols+k] * b.Data[k*b.Cols+j]
				}
				result.Data[row*result.Cols+j] = sum
			}
		}(i)
	}
	wg.Wait()
	return &result
}

// Multiplication of two matricies
func (m *Matrix) Dot(x, y *Matrix) {
	if x.Cols != y.Rows {
		panic("Incorrect matrix shapes for multiplication")
	}

	m.Rows = x.Rows
	m.Cols = y.Cols
	m.Data = make([]float64, x.Rows*y.Cols)

	for i := range x.Rows {
		for j := range y.Cols {
			sum := 0.0
			for k := 0; k < x.Cols; k++ {
				sum += x.Data[i*x.Cols+k] * y.Data[k*y.Cols+j]
			}
			m.Data[i*y.Cols+j] = sum
		}
	}
}

func DotNaive(x, y *Matrix) *Matrix {
	var m Matrix
	if x.Cols != y.Rows {
		panic("Incorrect matrix shapes for multiplication")
	}

	m.Rows = x.Rows
	m.Cols = y.Cols
	m.Data = make([]float64, x.Rows*y.Cols)

	for i := range x.Rows {
		for j := range y.Cols {
			sum := 0.0
			for k := 0; k < x.Cols; k++ {
				sum += x.Data[i*x.Cols+k] * y.Data[k*y.Cols+j]
			}
			m.Data[i*y.Cols+j] = sum
		}
	}

	return &m
}

// Adds a constant n to each element of a matrix
func (m *Matrix) AddElem(n float64) {
	for i := range m.Data {
		m.Data[i] += n
	}
}

// mutliples a constant n to each element of a matrix
func (m *Matrix) MultElem(n float64) {
	for i := range m.Data {
		m.Data[i] = m.Data[i] * n
	}
}

// sums together all the elements in a matrix
func (m *Matrix) Sum() float64 {
	sum := 0.0
	for i := range m.Data {
		sum += m.Data[i]
	}

	return sum
}

// Inverts a matrix using the Gauss-Jordan method
// First find the indentity matrix and then Gaussian elimination
func (m *Matrix) Inverse() error {

	if m.Rows != m.Cols {
		return ErrSquare
	}

	size := m.Rows
	data := make([]float64, size*size)
	copy(data, m.Data)

	// Create an I matrix
	I := make([]float64, size*size)
	for i := 0; i < size; i++ {
		I[i*size+i] = 1
	}

	// Gaussian elimination
	for col := 0; col < size; col++ {
		// Find pivot row
		maxRow := col
		for row := col + 1; row < size; row++ {
			if math.Abs(data[row*size+col]) > math.Abs(data[maxRow*size+col]) {
				maxRow = row
			}
		}

		// Swap rows
		if maxRow != col {
			// Swap data
			for k := 0; k < size; k++ {
				data[col*size+k], data[maxRow*size+k] = data[maxRow*size+k], data[col*size+k]
				I[col*size+k], I[maxRow*size+k] = I[maxRow*size+k], I[col*size+k]
			}
		}

		// Make diagonal element 1
		pivot := data[col*size+col]
		if pivot == 0 {
			return ErrSinguar
		}
		for k := 0; k < size; k++ {
			data[col*size+k] /= pivot
			I[col*size+k] /= pivot
		}

		// Make other elements in the column 0
		for row := 0; row < size; row++ {
			if row != col {
				factor := data[row*size+col]
				for k := 0; k < size; k++ {
					data[row*size+k] -= factor * data[col*size+k]
					I[row*size+k] -= factor * I[col*size+k]
				}
			}
		}
	}

	m.Rows = size
	m.Cols = size
	m.Data = I

	return nil

}

// Sees if two matricies are equal with a certain tolerance due to floating point issues
func ApproxEquals(x, y *Matrix, tol float64) bool {

	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic("Cannot compare matrices with different shapes")
	}

	for i := range x.Cols * x.Rows {
		if math.Abs(x.Data[i]-y.Data[i]) > tol {
			return false
		}
	}

	return true
}

func (m *Matrix) PrintMatrix() {

	maxWidth := 0
	for _, value := range m.Data {
		width := len(strconv.FormatFloat(value, 'f', 6, 64))
		if width > maxWidth {
			maxWidth = width
		}
	}

	for i := range m.Rows {

		switch {
		case m.Rows == 1:
			fmt.Print("[")
		case i == 0:
			fmt.Print("⎡")
		case i+1 == m.Rows:
			fmt.Print("⎣")
		default:
			fmt.Print("⎢")
		}

		for j := range m.Cols {
			value := strconv.FormatFloat(m.Data[i*m.Cols+j], 'f', 6, 64)
			paddedValue := fmt.Sprintf("%*s", maxWidth, value)
			if j+1 == m.Cols {
				fmt.Print(paddedValue)
			} else {
				fmt.Print(paddedValue, " ")
			}
		}

		switch {
		case m.Rows == 1:
			fmt.Print("]")
		case i == 0:
			fmt.Print("⎤")
		case i+1 == m.Rows:
			fmt.Print("⎦")
		default:
			fmt.Print("⎥")
		}

		fmt.Println()

	}
}

// Transposes a matrix and places the result in the receiver
func (m *Matrix) T() {

	transposed := Matrix{
		Rows: m.Cols,
		Cols: m.Rows,
		Data: make([]float64, m.Cols*m.Rows),
	}

	for i := range m.Rows {
		for j := range m.Cols {
			transposed.Data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}

	m.Rows = transposed.Rows
	m.Cols = transposed.Cols
	m.Data = transposed.Data

}

// Ramdomly shuffles the rows of the Matrix, needed for stochastic gradient descent
// Picks the current row and swaps with a random row in the matrix
// Takes the X and y matrix and shuffles such that they still correspond to each other
func ShuffleRows(X, y *Matrix) {
	for i, r := range rand.Perm(X.Rows) {

		//Shuffle the X data
		XcurrentRow := X.Data[X.Cols*i : X.Cols*i+X.Cols]
		XrandomRow := X.Data[X.Cols*r : X.Cols*r+X.Cols]
		for i := range XcurrentRow {
			XcurrentRow[i], XrandomRow[i] = XrandomRow[i], XcurrentRow[i]
		}

		//shuffle the y data
		yCurrentRow := y.Data[y.Cols*i : y.Cols*i+y.Cols]
		yRandomRow := y.Data[y.Cols*r : y.Cols*r+y.Cols]
		for i := range yCurrentRow {
			yCurrentRow[i], yRandomRow[i] = yRandomRow[i], yCurrentRow[i]
		}

	}
}
