{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RankerCore where

import Clash.Prelude

newtype Score32 = Score32 { unScore32 :: Unsigned 32 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype SegmentID64 = SegmentID64 { unSegmentID64 :: Unsigned 64 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype Position64 = Position64 { unPosition64 :: Unsigned 64 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype QueryHash64 = QueryHash64 { unQueryHash64 :: Unsigned 64 }
    deriving (Generic, NFDataX, Eq, Ord)

data RankRequest = RankRequest
    { queryHash :: QueryHash64
    , segmentID :: SegmentID64
    , segmentPos :: Position64
    , baseScore :: Score32
    } deriving (Generic, NFDataX, Eq)

data RankResult = RankResult
    { resultID :: SegmentID64
    , finalScore :: Score32
    , rank :: Unsigned 16
    } deriving (Generic, NFDataX, Eq)

data RankerState = RankerState
    { stateCounter :: Unsigned 16
    , lastScore :: Score32
    , lastQuery :: Maybe QueryHash64
    } deriving (Generic, NFDataX, Eq)

initialState :: RankerState
initialState = RankerState 0 (Score32 0) Nothing

positionBiasScale :: Unsigned 32
positionBiasScale = 1000

safeDiv :: Unsigned 32 -> Unsigned 32 -> Unsigned 32
safeDiv _ 0 = 0
safeDiv a b = a `div` b

computePositionBias :: Position64 -> Score32
computePositionBias pos = Score32 $ safeDiv positionBiasScale (resize (unPosition64 pos) + 1)

computeFinalScore :: Score32 -> Score32 -> Score32
computeFinalScore (Score32 base) (Score32 bias) = Score32 (base + bias)

mkRankRequest :: Unsigned 64 -> Unsigned 64 -> Unsigned 64 -> Unsigned 32 -> RankRequest
mkRankRequest qh sid pos base = RankRequest
    (QueryHash64 qh)
    (SegmentID64 sid)
    (Position64 pos)
    (Score32 base)

rankerCore
    :: HiddenClockResetEnable dom
    => Signal dom (Maybe RankRequest)
    -> Signal dom (Maybe RankResult)
rankerCore = mealy rankerT initialState

rankerT
    :: RankerState
    -> Maybe RankRequest
    -> (RankerState, Maybe RankResult)
rankerT state Nothing = (state, Nothing)
rankerT state (Just req) = (newState, Just result)
  where
    newCounter = case lastQuery state of
        Just prevQuery | prevQuery == queryHash req -> stateCounter state + 1
        _ -> 1

    bias = computePositionBias (segmentPos req)
    final = computeFinalScore (baseScore req) bias

    result = RankResult (segmentID req) final newCounter

    newState = RankerState newCounter final (Just (queryHash req))

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System (Maybe RankRequest)
    -> Signal System (Maybe RankResult)
topEntity = exposeClockResetEnable rankerCore

testRankRequest :: RankRequest
testRankRequest = mkRankRequest 0x123456789ABCDEF0 0xFEDCBA9876543210 10 1000

simulateRanker :: Maybe RankRequest -> (Unsigned 16, Score32)
simulateRanker Nothing = (0, Score32 0)
simulateRanker (Just req) =
    let bias = computePositionBias (segmentPos req)
        final = computeFinalScore (baseScore req) bias
    in (1, final)

================