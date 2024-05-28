module LinearAlgebra (matMul, matVecMul, vecAdd, outerProduct) where

import Data.List (transpose)
import Types

matMul :: (Num a) => Matrix a -> Matrix a -> Matrix a
matMul a b = [[sum $ zipWith (*) ar bc | bc <- transpose b] | ar <- a]

matVecMul :: (Num a) => Matrix a -> Vector a -> Vector a
matVecMul m v = map (sum . zipWith (*) v) (transpose m)

vecAdd :: (Num a) => Vector a -> Vector a -> Vector a
vecAdd = zipWith (+)

outerProduct :: (Num a) => Vector a -> Vector a -> Matrix a
outerProduct xs ys = [[x * y | y <- ys] | x <- xs]
