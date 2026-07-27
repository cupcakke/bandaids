{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NoImplicitPrelude #-}

module SSISearch where

import Clash.Prelude
import GHC.TypeLits

newtype HashKey64 = HashKey64 { unHashKey64 :: Unsigned 64 }
    deriving (Generic, NFDataX, Eq, Ord)

newtype NodeAddr32 = NodeAddr32 { unNodeAddr32 :: Unsigned 32 }
    deriving (Generic, NFDataX, Eq, Ord)

data SearchRequest = SearchRequest
    { searchKey :: HashKey64
    , rootAddr :: NodeAddr32
    } deriving (Generic, NFDataX, Eq)

data SearchResult = SearchResult
    { foundAddr :: NodeAddr32
    , found :: Bool
    , depth :: Unsigned 8
    } deriving (Generic, NFDataX, Eq)

data TreeNode = TreeNode
    { nodeKey :: HashKey64
    , leftChild :: NodeAddr32
    , rightChild :: NodeAddr32
    , isValid :: Bool
    } deriving (Generic, NFDataX, Eq)

data SearchState
    = Idle
    | Fetching NodeAddr32 (Unsigned 8)
    | Comparing NodeAddr32 (Unsigned 8) TreeNode
    deriving (Generic, NFDataX, Eq)

type MaxSearchDepthConfig = 64

maxSearchDepthVal :: Unsigned 8
maxSearchDepthVal = fromInteger (natVal (Proxy :: Proxy MaxSearchDepthConfig))

nullAddr :: NodeAddr32
nullAddr = NodeAddr32 0

mkSearchRequest :: Unsigned 64 -> Unsigned 32 -> SearchRequest
mkSearchRequest k r = SearchRequest (HashKey64 k) (NodeAddr32 r)

mkTreeNode :: Unsigned 64 -> Unsigned 32 -> Unsigned 32 -> Bool -> TreeNode
mkTreeNode k l r v = TreeNode (HashKey64 k) (NodeAddr32 l) (NodeAddr32 r) v

ssiSearch
    :: HiddenClockResetEnable dom
    => Signal dom (Maybe SearchRequest)
    -> Signal dom (Maybe TreeNode)
    -> (Signal dom (Maybe NodeAddr32), Signal dom (Maybe SearchResult))
ssiSearch reqIn nodeIn = (memReq, resultOut)
  where
    (memReq, resultOut) = unbundle $ mealy ssiSearchT Idle (bundle (reqIn, nodeIn))

ssiSearchT
    :: SearchState
    -> (Maybe SearchRequest, Maybe TreeNode)
    -> (SearchState, (Maybe NodeAddr32, Maybe SearchResult))
ssiSearchT Idle (Just req, _) =
    (Fetching (rootAddr req) 0, (Just (rootAddr req), Nothing))
ssiSearchT Idle _ = (Idle, (Nothing, Nothing))

ssiSearchT (Fetching addr currentDepth) (mReq, Just node)
    | currentDepth >= maxSearchDepthVal = (Idle, (Nothing, Just depthExceeded))
    | not (isValid node) = (Idle, (Nothing, Just notFound))
    | otherwise = case mReq of
        Just req -> checkNode req addr (currentDepth + 1) node
        Nothing -> (Comparing addr (currentDepth + 1) node, (Nothing, Nothing))
  where
    notFound = SearchResult nullAddr False currentDepth
    depthExceeded = SearchResult nullAddr False maxSearchDepthVal
ssiSearchT (Fetching addr currentDepth) _ = (Fetching addr currentDepth, (Nothing, Nothing))

ssiSearchT (Comparing addr currentDepth node) (Just req, _) =
    checkNode req addr currentDepth node
ssiSearchT (Comparing addr currentDepth node) _ = (Comparing addr currentDepth node, (Nothing, Nothing))

checkNode :: SearchRequest -> NodeAddr32 -> Unsigned 8 -> TreeNode -> (SearchState, (Maybe NodeAddr32, Maybe SearchResult))
checkNode req addr currentDepth node
    | currentDepth >= maxSearchDepthVal = (Idle, (Nothing, Just depthExceeded))
    | unHashKey64 (searchKey req) == unHashKey64 (nodeKey node) = (Idle, (Nothing, Just foundResult))
    | unHashKey64 (searchKey req) < unHashKey64 (nodeKey node) && leftChild node /= nullAddr =
        (Fetching (leftChild node) currentDepth, (Just (leftChild node), Nothing))
    | unHashKey64 (searchKey req) > unHashKey64 (nodeKey node) && rightChild node /= nullAddr =
        (Fetching (rightChild node) currentDepth, (Just (rightChild node), Nothing))
    | otherwise = (Idle, (Nothing, Just notFound))
  where
    foundResult = SearchResult addr True currentDepth
    notFound = SearchResult nullAddr False currentDepth
    depthExceeded = SearchResult nullAddr False maxSearchDepthVal

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System (Maybe SearchRequest)
    -> Signal System (Maybe TreeNode)
    -> (Signal System (Maybe NodeAddr32), Signal System (Maybe SearchResult))
topEntity = exposeClockResetEnable ssiSearch

testSearchRequest :: SearchRequest
testSearchRequest = mkSearchRequest 0x123456 0x1000

testTreeNode :: TreeNode
testTreeNode = mkTreeNode 0x123456 0x2000 0x3000 True

simulateSearch :: Maybe SearchRequest -> Maybe TreeNode -> (SearchState, Maybe SearchResult)
simulateSearch Nothing _ = (Idle, Nothing)
simulateSearch (Just req) Nothing = (Fetching (rootAddr req) 0, Nothing)
simulateSearch (Just req) (Just node)
    | not (isValid node) = (Idle, Just notFound)
    | unHashKey64 (searchKey req) == unHashKey64 (nodeKey node) = (Idle, Just foundRes)
    | unHashKey64 (searchKey req) < unHashKey64 (nodeKey node) = (Fetching (leftChild node) 1, Nothing)
    | otherwise = (Fetching (rightChild node) 1, Nothing)
  where
    notFound = SearchResult nullAddr False 0
    foundRes = SearchResult (rootAddr req) True 1

================