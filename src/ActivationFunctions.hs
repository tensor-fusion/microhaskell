module ActivationFunctions (sigmoid, applySigmoid) where

import AutoDiff
import Types

sigmoid :: (Floating a) => Dual a -> Dual a
sigmoid (Dual x dx) = Dual (1 / (1 + exp (- x))) (dx * exp (- x) / (1 + exp (- x)) ^ (2 :: Integer))

-- | Applies sigmoid to each element in a vector
applySigmoid :: (Eq a, Floating a) => Vector (Dual a) -> Vector (Dual a)
applySigmoid = map sigmoid
