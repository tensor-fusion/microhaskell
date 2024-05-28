module Types where

import AutoDiff

type Matrix a = [[a]]

type Vector a = [a]

-- | 2-layer feedforward net
data NeuralNet a = NeuralNet
  { w1 :: Matrix (Dual a),
    b1 :: Vector (Dual a),
    w2 :: Matrix (Dual a),
    b2 :: Vector (Dual a)
  }
  deriving (Eq, Show)
