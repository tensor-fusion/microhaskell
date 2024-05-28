{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-unused-matches #-}

module Backprop (backprop, updateParams) where

import AutoDiff
import Data.List (transpose)
import LinearAlgebra
import Types

updateParams :: NeuralNet Double -> NeuralNet Double -> Double -> NeuralNet Double
updateParams (NeuralNet w1 b1 w2 b2) (NeuralNet gw1 gb1 gw2 gb2) learningRate =
  NeuralNet (updateMatrix w1 gw1) (updateVector b1 gb1) (updateMatrix w2 gw2) (updateVector b2 gb2)
  where
    lrDual = makeDualConstant learningRate

    updateMatrix :: Matrix (Dual Double) -> Matrix (Dual Double) -> Matrix (Dual Double)
    updateVector :: Vector (Dual Double) -> Vector (Dual Double) -> Vector (Dual Double)
    updateMatrix = zipWith updateVector
    updateVector = zipWith (\o n -> o - lrDual * n)

-- | Computes param gradients via backprop
backprop :: NeuralNet Double -> Vector (Dual Double) -> Vector (Dual Double) -> Vector (Dual Double) -> Dual Double -> NeuralNet Double
backprop (NeuralNet w1 b1 w2 b2) inputs hiddenOutputs outputs lossGradient = NeuralNet gw1 gb1 gw2 gb2
  where
    outputErrors = map (* lossGradient) outputs

    hiddenOutputDerivatives = map (\o -> o * (1 - o)) hiddenOutputs
    gw2 = outerProduct hiddenOutputs outputErrors
    gb2 = outputErrors

    hiddenErrors = matVecMul (transpose w2) outputErrors
    gw1 = outerProduct inputs (zipWith (*) hiddenErrors hiddenOutputDerivatives)
    gb1 = zipWith (*) hiddenErrors hiddenOutputDerivatives
