{-# OPTIONS_GHC -Wno-name-shadowing #-}

module ForwardProp (forwardLayer, forwardProp) where

import ActivationFunctions
import AutoDiff
import LinearAlgebra
import Types

-- | Forward pass for a single layer
forwardLayer :: (Floating a, Eq a) => Matrix (Dual a) -> Vector (Dual a) -> Vector (Dual a) -> Vector (Dual a)
forwardLayer weights biases inputs = applySigmoid $ vecAdd (head (matMul [inputs] weights)) biases

-- | Forward pass for the entire net
forwardProp :: (Eq a, Floating a) => NeuralNet a -> Vector (Dual a) -> (Vector (Dual a), Vector (Dual a))
forwardProp (NeuralNet w1 b1 w2 b2) input = (hiddenOutputs, output)
  where
    hiddenOutputs = forwardLayer w1 b1 input
    output = forwardLayer w2 b2 hiddenOutputs
