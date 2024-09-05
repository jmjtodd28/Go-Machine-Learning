package utils

import (
	"reflect"
	"strconv"
	"testing"
)

func TestCreateMatrix(t *testing.T) {

	tests := []struct {
		d            []float64
		rows, cols   int
		mat          *Matrix
		shouldPanic  bool
		panicMessage string
	}{
		{
			[]float64{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0},
			3, 3,
			&Matrix{
				Rows: 3, Cols: 3,
				Data: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			},
			false, "",
		},
		{
			[]float64{
				1, 2, 3,
				4, 5, 6},
			2, 3,
			&Matrix{
				Rows: 2, Cols: 3,
				Data: []float64{1, 2, 3, 4, 5, 6},
			},
			false, "",
		},
		{
			[]float64{1, 2, 3},
			1, 4,
			nil,
			true, "Dimensions do not match the data",
		},
		{
			[]float64{1, 2, 3},
			2, 0,
			nil,
			true, "Cannot have a Matirx dimension = 0",
		},
		{
			[]float64{1, 2, 3},
			-3, -1,
			nil,
			true, "Cannot have negative dimensions",
		},
	}

	for i, test := range tests {
		t.Run("TestCase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				CreateMatrix(test.rows, test.cols, test.d)
			} else {
				m := CreateMatrix(test.rows, test.cols, test.d)
				rows := m.Rows
				cols := m.Cols
				if rows != test.rows {
					t.Errorf("unexpected number of rows for test %d: got: %d want %d", i, rows, test.rows)
				}
				if cols != test.cols {
					t.Errorf("unexpected number of columns for test %d: got: %d want %d", i, cols, test.cols)
				}
				if !reflect.DeepEqual(m, test.mat) {
					t.Errorf("Matrix Does not equal the expected Matrix in test %d", i)
				}
			}
		})
	}
}

func TestInvertMatrix(t *testing.T) {

	tests := []struct {
		testMat, expectedMat Matrix
		shouldPass           bool
		expectedError        error
	}{
		{
			Matrix{
				Rows: 3, Cols: 3,
				Data: []float64{
					3, 0, 2,
					2, 0, -2,
					0, 1, 1},
			},
			Matrix{
				Rows: 3, Cols: 3,
				Data: []float64{
					0.2, 0.2, 0,
					-0.2, 0.3, 1,
					0.2, -0.3, 0},
			},
			true,
			nil,
		}, {

			Matrix{
				Rows: 4, Cols: 4,
				Data: []float64{
					4, 0, 0, 0,
					0, 0, 2, 0,
					0, 1, 2, 0,
					1, 0, 0, 1},
			}, Matrix{
				Rows: 4, Cols: 4,
				Data: []float64{
					0.25, 0, 0, 0,
					0, -1, 1, 0,
					0, 0.5, 0, 0,
					-0.25, 0, 0, 1},
			},
			true,
			nil,
		},
		{
			Matrix{
				Rows: 2, Cols: 3,
				Data: []float64{
					1, 2, 3,
					4, 5, 6},
			},
			Matrix{},
			false,
			ErrSquare,
		},
		{
			Matrix{
				Rows: 2, Cols: 2,
				Data: []float64{4, 2, 12, 6},
			},
			Matrix{},
			false,
			ErrSinguar,
		},
	}

	for i, test := range tests {
		t.Run("TestCase"+strconv.Itoa(i), func(t *testing.T) {

			err := test.testMat.Inverse()

			//The test is supposed to pass but instead failed
			if test.shouldPass && err != nil {
				t.Fatalf("Test %d: was expected to pass but failed with an error: %s", i+1, err)
			}

			//The test is supposed to fail but instead passed
			if !test.shouldPass && err == nil {
				t.Fatalf("Test %d: was expected to fail but succeeded, should have got error: %s", i+1, test.expectedError)
			}

			//The test fails but need to check the error message
			if !test.shouldPass && err != nil {
				if test.expectedError != err {
					t.Fatalf("Test failed with the error: %s, but instead failed with %s", err, test.expectedError)
				}
			}

			//Compare the contents of the test and the expected inversed matrix
			//Cant do direct comparison due to floating point issues so there is a margin for error
			if test.shouldPass && err == nil {
				tol := 1e-9
				if !ApproxEquals(&test.testMat, &test.expectedMat, tol) {
					t.Fatalf("The test to see if the two matrices are equal with a tolerance %f failed, have:%v , got:%v", tol, test.testMat.Data, test.expectedMat.Data)
				}
			}
		})
	}
}

func TestMatrixAdd(t *testing.T) {
	//a + b = expected
	tests := []struct {
		a, b, expected *Matrix
		shouldPass     bool
		shouldPanic    bool
		panicMessage   string
	}{
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{2, 4, 6, 8, 10, 12, 14, 16, 18}},
			true,
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 1, 1, 1, 1, 1, 1, 1}},
			false,
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 2, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6}},
			nil,
			false,
			true,
			"Cannot add matricies with different shapes",
		}, {
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 2, Data: []float64{1, 2, 3, 4, 5, 6}},
			nil,
			false,
			true,
			"Cannot add matricies with different shapes",
		},
	}

	for i, test := range tests {

		t.Run("Testcase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				var have *Matrix
				have.Add(test.a, test.b)

			} else {
				var have Matrix
				have.Add(test.a, test.b)
				if test.shouldPass && !reflect.DeepEqual(have, *test.expected) {
					t.Errorf("Matrix != expected matrix in test %d: have: %v, expected %v", i, have, test.expected)
				}

				if !test.shouldPass && reflect.DeepEqual(have, *test.expected) {
					t.Errorf("Matrix should not be equal to expected but it is")
				}
			}
		})
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		a, b, expected *Matrix
		shouldPass     bool
		shouldPanic    bool
		panicMessage   string
	}{
		{
			&Matrix{Rows: 2, Cols: 2, Data: []float64{1, 2, 3, 4}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{5, 6, 7, 8}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{19, 22, 43, 50}}, // Resulting matrix
			true,
			false,
			"",
		},
		{
			&Matrix{Rows: 2, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{7, 8, 9, 10}},
			nil,
			false,
			true,
			"Incorrect matrix shapes for multiplication",
		},
		{
			&Matrix{Rows: 3, Cols: 2, Data: []float64{1, 4, 2, 5, 3, 6}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{7, 8, 9, 10}},
			&Matrix{Rows: 3, Cols: 2, Data: []float64{43, 48, 59, 66, 75, 84}}, // Resulting matrix
			true,
			false,
			"",
		},
		{
			&Matrix{Rows: 2, Cols: 2, Data: []float64{1, 2, 3, 4}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{5, 6, 7, 8}},
			&Matrix{Rows: 2, Cols: 2, Data: []float64{20, 22, 43, 50}}, // Incorrect expected matrix
			false,
			false,
			"",
		},
	}

	for i, test := range tests {
		t.Run("Testcase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("Test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				var have *Matrix
				have = Dot(test.a, test.b)
				_ = have

			} else {
				have := Dot(test.a, test.b)
				//if test.shouldPass && !reflect.DeepEqual(have, *test.expected) {
				for i := range have.Data {
					if have.Data[i] != test.expected.Data[i] && test.shouldPass {

						t.Errorf("Test %d: Matrix != expected matrix: have: %v, expected %v", i, have, test.expected)
					}
				}

				if !test.shouldPass && reflect.DeepEqual(have, *test.expected) {
					t.Errorf("Test %d: Matrix should not be equal to expected but it is", i)
				}
			}
		})
	}
}

func TestMultiply(t *testing.T) {
	tests := []struct {
		a, b, expected *Matrix
		shouldPass     bool
		shouldPanic    bool
		panicMessage   string
	}{
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 4, 9, 16, 25, 36, 49, 64, 81}},
			true,
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 1, 1, 1, 1, 1, 1, 1}},
			false,
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 2, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6}},
			nil,
			false,
			true,
			"Cannot multiply matricies with different shapes",
		}, {
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			&Matrix{Rows: 3, Cols: 2, Data: []float64{1, 2, 3, 4, 5, 6}},
			nil,
			false,
			true,
			"Cannot multiply matricies with different shapes",
		},
	}
	for i, test := range tests {

		t.Run("Testcase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				var have *Matrix
				have.Multiply(test.a, test.b)

			} else {
				var have Matrix
				have.Multiply(test.a, test.b)
				if test.shouldPass && !reflect.DeepEqual(have, *test.expected) {
					t.Errorf("Matrix != expected matrix in test %d: have: %v, expected %v", i, have, test.expected)
				}

				if !test.shouldPass && reflect.DeepEqual(have, *test.expected) {
					t.Errorf("Matrix should not be equal to expected but it is")
				}
			}
		})
	}

}

func TestCol(t *testing.T) {
	tests := []struct {
		m            *Matrix
		n            int
		expected     *Matrix
		shouldPanic  bool
		panicMessage string
	}{
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			0,
			&Matrix{Rows: 3, Cols: 1, Data: []float64{1, 4, 7}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			1,
			&Matrix{Rows: 3, Cols: 1, Data: []float64{2, 5, 8}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			2,
			&Matrix{Rows: 3, Cols: 1, Data: []float64{3, 6, 9}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			3,
			nil,
			true,
			"Column index out of range",
		}, {
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			-1,
			nil,
			true,
			"Cannot have negative column index",
		},
	}

	for i, test := range tests {
		t.Run("TestCase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				test.m.Col(test.n)
			} else {
				have := test.m.Col(test.n)
				if !reflect.DeepEqual(have, test.expected) {
					t.Errorf("Matrix != expected matrix in test %d: have: %v, expected %v", i, have, test.expected)
				}
			}
		})
	}

}

func TestRow(t *testing.T) {
	tests := []struct {
		m            *Matrix
		n            int
		expected     *Matrix
		shouldPanic  bool
		panicMessage string
	}{
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			0,
			&Matrix{Rows: 1, Cols: 3, Data: []float64{1, 2, 3}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			1,
			&Matrix{Rows: 1, Cols: 3, Data: []float64{4, 5, 6}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			2,
			&Matrix{Rows: 1, Cols: 3, Data: []float64{7, 8, 9}},
			false,
			"",
		},
		{
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			3,
			nil,
			true,
			"Row index out of range",
		}, {
			&Matrix{Rows: 3, Cols: 3, Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			-1,
			nil,
			true,
			"Cannot have negative row index",
		},
	}

	for i, test := range tests {
		t.Run("TestCase"+strconv.Itoa(i), func(t *testing.T) {
			if test.shouldPanic {
				defer func() {
					if r := recover(); r != nil {
						if r != test.panicMessage {
							t.Errorf("test %d: unexpected panic message: got %v, want %v", i, r, test.panicMessage)
						}
					} else {
						t.Errorf("Test %d: expected panic but did not panic", i)
					}
				}()
				test.m.Row(test.n)
			} else {
				have := test.m.Row(test.n)
				if !reflect.DeepEqual(have, test.expected) {
					t.Errorf("Matrix != expected matrix in test %d: have: %v, expected %v", i, have, test.expected)
				}
			}
		})
	}

}
