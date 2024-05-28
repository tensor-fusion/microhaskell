-- This is the tiniest possible version of an autodiff library with dual types.
-- Based on this excellent blog post: https://www.danielbrice.net/blog/automatic-differentiation-is-trivial-in-haskell
{-# OPTIONS_GHC -Wno-unused-matches #-}

module AutoDiff where

-- | The Dual type represents a number and its derivative
data Dual a = Dual a a
  deriving (Eq, Read, Show)

-- | Instance of Num typeclass for Dual numbers. Defines how to perform arithmetic ops on Dual numbers
instance Num a => Num (Dual a) where
  (Dual u u') + (Dual v v') = Dual (u + v) (u' + v')
  (Dual u u') * (Dual v v') = Dual (u * v) (u' * v + u * v')
  (Dual u u') - (Dual v v') = Dual (u - v) (u' - v')
  abs (Dual u u') = Dual (abs u) (u' * signum u)
  signum (Dual u u') = Dual (signum u) 0
  fromInteger n = Dual (fromInteger n) 0

-- | Convert a constant to a Dual number with a derivative of zero
makeDualConstant :: Num a => a -> Dual a
makeDualConstant x = Dual x 0

-- | Create a Dual number with a given value and derivative
makeDualVariable :: a -> a -> Dual a
makeDualVariable = Dual

-- | Extract value from Dual number
getDualValue :: Dual a -> a
getDualValue (Dual x _) = x

-- | Extract derivative from Dual number
getDualDerivative :: Dual a -> a
getDualDerivative (Dual _ x') = x'

-- | Differentiate a single-variable function `f`
differentiate :: Num a => (Dual a -> Dual c) -> a -> c
differentiate f x = getDualDerivative . f $ Dual x 1