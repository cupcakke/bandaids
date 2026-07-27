{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}

module MemoryArbiter where

import Clash.Prelude

newtype Addr32 = Addr32 { unAddr32 :: Unsigned 32 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype Data64 = Data64 { unData64 :: Unsigned 64 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype ClientID4 = ClientID4 { unClientID4 :: Unsigned 4 }
    deriving (Generic, NFDataX, Eq, Ord)

type NumClients = 4
type ServiceCycles = 4

data MemRequest = MemRequest
    { reqAddr :: Addr32
    , reqWrite :: Bool
    , reqData :: Data64
    , reqClient :: ClientID4
    } deriving (Generic, NFDataX, Eq)

data MemResponse = MemResponse
    { respData :: Data64
    , respClient :: ClientID4
    , respValid :: Bool
    } deriving (Generic, NFDataX, Eq)

data ArbiterState
    = ArbIdle
    | ArbServing ClientID4 (Unsigned 8)
    deriving (Generic, NFDataX, Eq)

mkMemRequest :: Unsigned 32 -> Bool -> Unsigned 64 -> Unsigned 4 -> MemRequest
mkMemRequest a w d c = MemRequest (Addr32 a) w (Data64 d) (ClientID4 c)

mkMemResponse :: Unsigned 64 -> Unsigned 4 -> Bool -> MemResponse
mkMemResponse d c v = MemResponse (Data64 d) (ClientID4 c) v

memoryArbiter
    :: HiddenClockResetEnable dom
    => Vec NumClients (Signal dom (Maybe MemRequest))
    -> Signal dom (Maybe MemResponse)
    -> (Signal dom (Maybe MemRequest), Vec NumClients (Signal dom (Maybe MemResponse)))
memoryArbiter clientReqs memResp = (memReqOut, clientResps)
  where
    (memReqOut, _grantVec) = unbundle $ mealy arbiterT (ArbIdle, 0) (bundle clientReqs)
    clientResps = map (\i -> fmap (filterResp (ClientID4 i)) memResp) (iterateI (+1) 0)

filterResp :: ClientID4 -> MemResponse -> Maybe MemResponse
filterResp cid resp
    | respClient resp == cid && respValid resp = Just resp
    | otherwise = Nothing

arbiterT
    :: (ArbiterState, Unsigned 8)
    -> Vec NumClients (Maybe MemRequest)
    -> ((ArbiterState, Unsigned 8), (Maybe MemRequest, Vec NumClients Bool))
arbiterT (ArbIdle, counter) reqs = case findIndex isJust reqs of
    Just idx ->
        let clientId = ClientID4 (resize (pack idx))
            grant = map (\i -> i == idx) (iterateI (+1) 0)
        in ((ArbServing clientId 0, counter + 1), (reqs !! idx, grant))
    Nothing -> ((ArbIdle, counter), (Nothing, repeat False))

arbiterT (ArbServing client cycles, counter) _reqs
    | cycles < fromInteger (natVal (Proxy :: Proxy ServiceCycles)) - 1 =
        ((ArbServing client (cycles + 1), counter), (Nothing, repeat False))
    | otherwise = ((ArbIdle, counter), (Nothing, repeat False))

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Vec NumClients (Signal System (Maybe MemRequest))
    -> Signal System (Maybe MemResponse)
    -> (Signal System (Maybe MemRequest), Vec NumClients (Signal System (Maybe MemResponse)))
topEntity = exposeClockResetEnable memoryArbiter

testInput :: Vec NumClients (Signal System (Maybe MemRequest))
testInput =
    ( pure (Just (mkMemRequest 0x1000 False 0 0))
    :> pure (Just (mkMemRequest 0x2000 True 0xDEADBEEF 1))
    :> pure Nothing
    :> pure Nothing
    :> Nil
    )

================