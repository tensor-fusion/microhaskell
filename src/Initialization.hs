module Initialization (initVector, initMatrix) where

import AutoDiff
import Data.List (foldl')
import System.Random
import Types

-- | Init vector with random elements
initVector :: (RandomGen g, Random a, Fractional a) => g -> Int -> (Vector (Dual a), g)
initVector gen size = foldl' generateElem ([], gen) [1 .. size]
  where
    generateElem (acc, g) _ =
      let (val, newG) = randomR (-0.5, 0.5) g
       in (acc ++ [makeDualConstant val], newG)

-- | Init matrix with random elements
initMatrix :: (RandomGen g, Random a, Fractional a) => g -> Int -> Int -> (Matrix (Dual a), g)
initMatrix gen rows cols = foldl' generateRow ([], gen) [1 .. rows]
  where
    generateRow (acc, g) _ =
      let (row, newG) = initVector g cols
       in (acc ++ [row], newG)
