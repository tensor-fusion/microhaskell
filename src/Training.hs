{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Training (trainEpoch) where

import AutoDiff
import Backprop (backprop, updateParams)
import Data.List
import ForwardProp (forwardProp)
import LossFunctions (lossGradient, lossMSE)
import Types

trainEpoch :: Double -> [([Double], Double)] -> (NeuralNet Double, [Double]) -> Int -> IO (NeuralNet Double, [Double])
trainEpoch learningRate dataSet (net, losses) _ = do
  let (net', epochLoss) = foldl' updateNetwork (net, 0) dataSet
  return (net', losses ++ [epochLoss / fromIntegral (length dataSet)])
  where
    updateNetwork (net, accumLoss) (inputs, trueOutput) =
      let inputsDual = map (`makeDualVariable` 1) inputs
          (hiddenOutputs, outputs) = forwardProp net inputsDual
          loss = lossMSE (head outputs) (makeDualConstant trueOutput)
          gradients = backprop net inputsDual hiddenOutputs outputs (lossGradient (head outputs) trueOutput)
          updatedNet = updateParams net gradients learningRate
       in (updatedNet, accumLoss + getDualValue loss)
