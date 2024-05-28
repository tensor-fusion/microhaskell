module LossFunctions (lossMSE, lossGradient) where

import AutoDiff

lossMSE :: Dual Double -> Dual Double -> Dual Double
lossMSE output target = (output - target) ^ (2 :: Integer)

lossGradient :: Dual Double -> Double -> Dual Double
lossGradient output target = 2 * (output - makeDualConstant target)
