module NeuralNet
  ( NeuralNet (..),
    Matrix,
    Vector,
    initVector,
    initMatrix,
    matMul,
    matVecMul,
    vecAdd,
    outerProduct,
    sigmoid,
    applySigmoid,
    forwardLayer,
    forwardProp,
    backprop,
    updateParams,
    lossMSE,
    lossGradient,
    trainEpoch,
  )
where

import ActivationFunctions (applySigmoid, sigmoid)
import Backprop (backprop, updateParams)
import ForwardProp (forwardLayer, forwardProp)
import Initialization (initMatrix, initVector)
import LinearAlgebra (matMul, matVecMul, outerProduct, vecAdd)
import LossFunctions (lossGradient, lossMSE)
import Training (trainEpoch)
import Types (Matrix, NeuralNet (..), Vector)
