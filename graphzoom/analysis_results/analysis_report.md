# Comprehensive LAMG vs CMG++ Analysis Report
============================================================

## Executive Summary

**Key Finding**: LAMG maintains better connectivity (3.6 vs 78.0 avg components)

## Detailed Results

| Method | Config | Nodes | Edges | Reduction | Components | Largest Component |
|--------|--------|-------|-------|-----------|------------|-------------------|
| LAMG | Ratio 2 | 1169 | 3938 | 2.32x | 6 | 1066 |
| LAMG | Ratio 3 | 519 | 1985 | 5.22x | 3 | 475 |
| LAMG | Ratio 4 | 519 | 1985 | 5.22x | 3 | 475 |
| LAMG | Ratio 5 | 519 | 1985 | 5.22x | 3 | 475 |
| LAMG | Ratio 6 | 218 | 979 | 12.42x | 3 | 204 |
| CMG++ | Level 1 (k=10, d=15) | 941 | 2616 | 2.88x | 78 | 772 |
| CMG++ | Level 2 (k=10, d=15) | 398 | 1329 | 6.80x | 78 | 236 |
| CMG++ | Level 3 (k=10, d=15) | 253 | 794 | 10.70x | 78 | 92 |

## Connectivity Analysis

### LAMG Results
- **Ratio 2**: 6 components, largest = 91.2% of nodes
- **Ratio 3**: 3 components, largest = 91.5% of nodes
- **Ratio 4**: 3 components, largest = 91.5% of nodes
- **Ratio 5**: 3 components, largest = 91.5% of nodes
- **Ratio 6**: 3 components, largest = 93.6% of nodes

### CMG++ Results
- **Level 1 (k=10, d=15)**: 78 components, largest = 82.0% of nodes
- **Level 2 (k=10, d=15)**: 78 components, largest = 59.3% of nodes
- **Level 3 (k=10, d=15)**: 78 components, largest = 36.4% of nodes

## Implications for Graph Neural Networks

- **Connected graphs** enable global information flow
- **Disconnected components** limit message passing scope
- **Fewer components** generally lead to better GNN performance
- **Large dominant components** preserve most graph structure
