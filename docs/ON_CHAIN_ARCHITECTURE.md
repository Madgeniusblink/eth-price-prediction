# On-Chain Architecture for ETH Price Prediction Oracle

This document outlines the architecture for bringing the ETH price prediction system on-chain, enabling trustless, decentralized trading signal execution.

## Overview

The on-chain implementation transforms our prediction system into a decentralized oracle network that:
- Publishes price predictions and trading signals on-chain
- Enables automated trading execution via smart contracts
- Provides verifiable prediction accuracy tracking
- Integrates with DeFi protocols for trustless execution

## Architecture Components

### 1. Oracle Smart Contract (Solidity)

The core oracle contract stores predictions and provides an interface for consumers.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/AutomationCompatible.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ETHPricePredictionOracle
 * @notice Stores and serves ETH price predictions from off-chain ML model
 */
contract ETHPricePredictionOracle is AutomationCompatibleInterface, Ownable {

    struct Prediction {
        uint256 timestamp;
        uint256 currentPrice;
        uint256 predicted15m;
        uint256 predicted30m;
        uint256 predicted60m;
        uint256 predicted120m;
        uint8 signal; // 0=WAIT, 1=BUY, 2=SELL, 3=SHORT
        uint8 confidence; // 0=LOW, 1=MEDIUM, 2=HIGH
        int256 fundingRate; // scaled by 1e6
        uint256 fearGreedIndex;
        uint256 riskReward; // scaled by 1e2 (e.g., 150 = 1.5)
    }

    struct AccuracyRecord {
        uint256 predictionId;
        uint256 actualPrice;
        int256 errorPct; // scaled by 1e4 (e.g., 100 = 1%)
        bool directionCorrect;
    }

    // Prediction storage
    mapping(uint256 => Prediction) public predictions;
    mapping(uint256 => AccuracyRecord) public accuracyRecords;
    uint256 public latestPredictionId;
    uint256 public latestAccuracyId;

    // Access control
    address public oracleUpdater;

    // Events
    event PredictionUpdated(
        uint256 indexed predictionId,
        uint256 timestamp,
        uint256 currentPrice,
        uint8 signal
    );

    event AccuracyRecorded(
        uint256 indexed accuracyId,
        uint256 indexed predictionId,
        int256 errorPct,
        bool directionCorrect
    );

    // Modifiers
    modifier onlyUpdater() {
        require(msg.sender == oracleUpdater, "Only updater");
        _;
    }

    constructor(address _updater) {
        oracleUpdater = _updater;
    }

    /**
     * @notice Update oracle with new prediction
     * @dev Called by authorized off-chain service
     */
    function updatePrediction(
        uint256 currentPrice,
        uint256 predicted15m,
        uint256 predicted30m,
        uint256 predicted60m,
        uint256 predicted120m,
        uint8 signal,
        uint8 confidence,
        int256 fundingRate,
        uint256 fearGreedIndex,
        uint256 riskReward
    ) external onlyUpdater {
        latestPredictionId++;

        predictions[latestPredictionId] = Prediction({
            timestamp: block.timestamp,
            currentPrice: currentPrice,
            predicted15m: predicted15m,
            predicted30m: predicted30m,
            predicted60m: predicted60m,
            predicted120m: predicted120m,
            signal: signal,
            confidence: confidence,
            fundingRate: fundingRate,
            fearGreedIndex: fearGreedIndex,
            riskReward: riskReward
        });

        emit PredictionUpdated(
            latestPredictionId,
            block.timestamp,
            currentPrice,
            signal
        );
    }

    /**
     * @notice Get latest prediction data
     */
    function getLatestPrediction() external view returns (Prediction memory) {
        return predictions[latestPredictionId];
    }

    /**
     * @notice Record accuracy for past prediction
     */
    function recordAccuracy(
        uint256 predictionId,
        uint256 actualPrice,
        int256 errorPct,
        bool directionCorrect
    ) external onlyUpdater {
        latestAccuracyId++;

        accuracyRecords[latestAccuracyId] = AccuracyRecord({
            predictionId: predictionId,
            actualPrice: actualPrice,
            errorPct: errorPct,
            directionCorrect: directionCorrect
        });

        emit AccuracyRecorded(
            latestAccuracyId,
            predictionId,
            errorPct,
            directionCorrect
        );
    }

    /**
     * @notice Chainlink Automation: Check if upkeep is needed
     */
    function checkUpkeep(bytes calldata)
        external
        view
        override
        returns (bool upkeepNeeded, bytes memory)
    {
        // Trigger update every 15 minutes
        upkeepNeeded = (block.timestamp - predictions[latestPredictionId].timestamp) >= 900;
    }

    /**
     * @notice Chainlink Automation: Perform upkeep
     */
    function performUpkeep(bytes calldata) external override {
        // Trigger off-chain prediction update via event
        emit PredictionUpdateRequested(block.timestamp);
    }

    event PredictionUpdateRequested(uint256 timestamp);
}
```

### 2. Chainlink Automation Integration

**Purpose**: Automate prediction updates every 15 minutes without manual intervention.

**Implementation**:
- Deploy Chainlink Automation-compatible `checkUpkeep` and `performUpkeep` functions
- Register contract with Chainlink Automation registry
- Fund upkeep with LINK tokens

**Benefits**:
- Decentralized scheduling (no centralized cron jobs)
- Highly reliable execution
- Gas-efficient automation

### 3. Trading Execution Layer (GMX Integration on Arbitrum)

**Why GMX on Arbitrum**:
- Low gas costs compared to Ethereum mainnet
- Deep liquidity for ETH perpetuals
- Decentralized perpetual trading
- Proven track record and security

**Architecture**:

```solidity
/**
 * @title AutomatedTradingExecutor
 * @notice Executes trades on GMX based on oracle signals
 */
contract AutomatedTradingExecutor {

    ETHPricePredictionOracle public oracle;
    IGMXRouter public gmxRouter;

    // Risk parameters
    uint256 public maxPositionSize = 10 ether; // Max 10 ETH per trade
    uint256 public minRiskReward = 150; // Minimum 1.5 R:R
    uint8 public minConfidence = 1; // Minimum MEDIUM confidence

    struct Position {
        bool isOpen;
        uint256 size;
        uint256 entryPrice;
        uint256 stopLoss;
        uint256 takeProfit;
        uint8 positionType; // 1=LONG, 3=SHORT
    }

    Position public currentPosition;

    /**
     * @notice Execute trade based on oracle signal
     */
    function executeSignal() external {
        ETHPricePredictionOracle.Prediction memory pred = oracle.getLatestPrediction();

        // Safety checks
        require(pred.confidence >= minConfidence, "Confidence too low");
        require(pred.riskReward >= minRiskReward, "R:R too low");
        require(pred.signal != 0, "No signal");

        // Close existing position if signal changes
        if (currentPosition.isOpen && currentPosition.positionType != pred.signal) {
            _closePosition();
        }

        // Open new position
        if (!currentPosition.isOpen && (pred.signal == 1 || pred.signal == 3)) {
            _openPosition(pred);
        }
    }

    function _openPosition(ETHPricePredictionOracle.Prediction memory pred) internal {
        // Calculate position size based on risk parameters
        uint256 positionSize = _calculatePositionSize(pred.riskReward);

        // Execute trade on GMX
        // ... GMX integration code ...

        currentPosition = Position({
            isOpen: true,
            size: positionSize,
            entryPrice: pred.currentPrice,
            stopLoss: _calculateStopLoss(pred),
            takeProfit: _calculateTakeProfit(pred),
            positionType: pred.signal
        });
    }

    function _closePosition() internal {
        // Close position on GMX
        // ... GMX integration code ...

        currentPosition.isOpen = false;
    }
}
```

### 4. FabricBloc Bridge Integration

**Purpose**: Enable cross-chain deployment and liquidity access.

**Architecture**:
- Deploy oracle on multiple chains (Ethereum, Arbitrum, Base, Optimism)
- Use FabricBloc bridge for state synchronization
- Aggregate liquidity across chains

**Benefits**:
- Access to best execution prices across chains
- Reduced gas costs on L2s
- Enhanced decentralization

### 5. EIP-712 Typed Data Signing

**Purpose**: Secure, user-friendly transaction signing for off-chain order creation.

**Implementation**:

```javascript
const domain = {
  name: 'ETH Prediction Oracle',
  version: '1',
  chainId: 42161, // Arbitrum
  verifyingContract: oracleAddress
};

const types = {
  PredictionUpdate: [
    { name: 'currentPrice', type: 'uint256' },
    { name: 'predicted15m', type: 'uint256' },
    { name: 'signal', type: 'uint8' },
    { name: 'confidence', type: 'uint8' },
    { name: 'timestamp', type: 'uint256' }
  ]
};

// Sign prediction update
const signature = await signer._signTypedData(domain, types, predictionData);
```

**Benefits**:
- Human-readable transaction data
- Protection against phishing
- Standardized signing format

### 6. EIP-4337 Account Abstraction

**Purpose**: Gasless transactions and improved UX for users.

**Implementation**:
- Deploy UserOperation contracts for prediction consumers
- Implement paymaster for sponsored gas
- Enable batch transactions

**User Flow**:
1. User approves prediction oracle access (one-time)
2. Paymaster covers gas for prediction updates
3. Automated execution without user intervention

**Benefits**:
- Users don't need ETH for gas
- Improved onboarding experience
- Batch prediction updates and executions

## Risk Parameters & Safety Features

### Position Sizing
- **Max Position Size**: 10 ETH per trade
- **Max Account Risk**: 2% per trade
- **Max Daily Loss**: 5% of account

### Signal Filters
- **Minimum Risk:Reward**: 1.5
- **Minimum Confidence**: MEDIUM
- **Max Funding Rate**: ±0.01% (avoid extreme funding)
- **Fear & Greed Bounds**: 10-90 (avoid extreme sentiment)

### Circuit Breakers
- **Max Consecutive Losses**: 3 trades
- **Max Daily Trades**: 10 trades
- **Pause on High Volatility**: ATR > 2x average

### Slippage Protection
- **Max Slippage**: 0.5% on entry/exit
- **Use Limit Orders**: For non-urgent executions
- **Price Impact Checks**: Before large trades

## Deployment Strategy

### Phase 1: Testnet Deployment (Arbitrum Goerli)
- Deploy oracle contract
- Test Chainlink Automation
- Verify prediction accuracy
- Test GMX integration with paper trading

### Phase 2: Mainnet Oracle Deployment (Arbitrum)
- Deploy production oracle
- Fund Chainlink Automation
- Set conservative risk parameters
- Monitor for 2 weeks

### Phase 3: Trading Executor Deployment
- Deploy automated trading contract
- Start with small position sizes (0.1 ETH)
- Gradually increase limits based on performance
- Monitor and adjust risk parameters

### Phase 4: Multi-Chain Expansion
- Deploy on Ethereum mainnet
- Deploy on Base and Optimism
- Integrate FabricBloc bridge
- Enable cross-chain liquidity

## Security Considerations

### Smart Contract Security
- **Audits**: Full audit by reputable firm (Trail of Bits, OpenZeppelin)
- **Time Locks**: 48-hour delay for critical parameter changes
- **Multi-sig Control**: 3-of-5 multi-sig for oracle updater role
- **Emergency Pause**: Owner can pause in case of exploit

### Oracle Security
- **Data Source Diversity**: Use multiple price feeds (Chainlink, Pyth, custom)
- **Anomaly Detection**: Reject predictions with extreme deviations
- **Rate Limiting**: Max 1 update per 15 minutes
- **Signature Verification**: All updates must be signed by authorized key

### Operational Security
- **Key Management**: Hardware wallets for all critical keys
- **Monitoring**: 24/7 monitoring with alerts
- **Incident Response**: Documented playbook for exploits
- **Insurance**: Protocol insurance via Nexus Mutual

## Gas Optimization

- **Packed Structs**: Use uint8 where possible to save storage
- **Events vs Storage**: Use events for historical data
- **Batch Updates**: Group multiple operations
- **L2 Deployment**: Significantly cheaper gas on Arbitrum

## Future Enhancements

1. **Multi-Asset Support**: Extend to BTC, SOL, and other assets
2. **Ensemble Oracles**: Aggregate predictions from multiple models
3. **On-Chain ML**: Explore on-chain inference (Giza, Modulus)
4. **DAO Governance**: Community-controlled risk parameters
5. **Prediction Markets**: Enable users to bet on prediction accuracy
6. **NFT Credentials**: Mint accuracy NFTs for top performers

## Conclusion

This architecture provides a robust, decentralized foundation for bringing ML-based price predictions on-chain. By leveraging Chainlink Automation, GMX for execution, and modern Ethereum standards (EIP-712, EIP-4337), we create a trustless system that can execute trading strategies autonomously while maintaining strict risk controls.

The phased deployment strategy ensures we can iterate safely, and the comprehensive risk parameters protect users from excessive losses. Integration with FabricBloc enables cross-chain expansion and enhanced liquidity access.
