language: PYTHON
name:     "optimization"

variable {
    name: "temperature"
    type: FLOAT
    size: 1
    min: 2.0
    max: 15.0
}

variable {
    name: "lambda_teacher"
    type: FLOAT
    size: 1
    min: 0.20
    max: 0.90
}