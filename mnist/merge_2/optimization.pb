language: PYTHON
name:     "optimization"

variable {
    name: "temperature"
    type: FLOAT
    size: 1
    min: 1.0
    max: 20.0
}

variable {
    name: "lambda_teacher"
    type: FLOAT
    size: 1
    min: 0.00
    max: 1.00
}