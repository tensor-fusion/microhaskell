{-# OPTIONS_GHC -Wno-type-defaults #-}

module Main where

import AutoDiff
import Control.Monad (foldM, forM_)
import ForwardProp
import GHC.Base (when)
import Initialization
import NeuralNet
import System.Random
import Text.Printf

trainLoop :: NeuralNet Double -> [([Double], Double)] -> Double -> Int -> IO ()
trainLoop network dataSet learningRate epochs = do
  (finalNetwork, losses) <- foldM (trainEpoch learningRate dataSet) (network, []) [1 .. epochs]
  forM_ (zip [1 ..] losses) $ \(epoch, loss) ->
    when (epoch `mod` 1000 == 0) $
      printf "Epoch: %d\tLoss: %.8f\n" (epoch :: Int) loss
  putStrLn "\nReal\t\tPredicted"
  forM_ dataSet $ \(inputs, trueOutput) -> do
    let inputsDual = map (`makeDualVariable` 1) inputs
        (_, outputs) = forwardProp finalNetwork inputsDual
        prediction = getDualValue (head outputs)
    if trueOutput == fromIntegral (floor trueOutput)
      then printf "%d\t\t%.8f\n" (floor trueOutput :: Int) prediction
      else printf "%.8f\t%.8f\n" trueOutput prediction

xorDataSet :: [([Double], Double)]
xorDataSet = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

initXorNet :: IO (NeuralNet Double)
initXorNet = do
  gen <- newStdGen
  let (weights1, gen1) = initMatrix gen 2 2
  let (biases1, gen2) = initVector gen1 2
  let (weights2, gen3) = initMatrix gen2 2 1
  let (biases2, _) = initVector gen3 1
  return $ NeuralNet weights1 biases1 weights2 biases2

main :: IO ()
main = do
  xorNet <- initXorNet
  let learningRate = 0.1
  let epochs = 10000
  trainLoop xorNet xorDataSet learningRate epochs
  putStrLn "\nTraining complete\n"
