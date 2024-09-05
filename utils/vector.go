package utils

type Vector struct {
	Length int
	Data   []float64
}

func CreateVector(l int, data []float64) *Vector {
	if l != len(data) {
		panic("Specified length does not match up to data length")
	}

	if l == 0 {
		panic("Cannot have data of length zero")
	}

	if l < 0 {
		panic("cannot have a negative length")
	}

	return &Vector{Length: l, Data: data}
}

// Dot product of two vectors
func VecDot(x, y *Vector) float64 {

	if x.Length != y.Length {
		panic("Different vector lengths")
	}

	total := 0.0
	for i := range x.Data {
		total += x.Data[i] * y.Data[i]
	}
	return total
}

// Element-wise addition of two vectors
func (v *Vector) Add(x, y *Vector) {
	if x.Length != y.Length {
		panic("Cannot add vectors of different lenghts")
	}

	v.Data = make([]float64, x.Length)

	for i := range x.Length {
		v.Data[i] = x.Data[i] + y.Data[i]
	}

	v.Length = x.Length
}
