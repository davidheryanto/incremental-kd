language: PYTHON
name:     "optimization"

variable {
    name: "temperature"
    type: FLOAT
    size: 1
    min: 2.0
    max: 10.0
}

variable {
    name: "lambda_teacher"
    type: FLOAT
    size: 1
    min: 0.25
    max: 0.80
}