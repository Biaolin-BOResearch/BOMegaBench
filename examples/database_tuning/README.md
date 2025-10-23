# Database Tuning Examples

This directory contains examples and documentation for using BenchBase with BOMegaBench's database tuning module.

## Files

### Setup Guides

- **[QUICKSTART.md](QUICKSTART.md)**: Get started in 15 minutes
- **[BENCHBASE_SETUP.md](BENCHBASE_SETUP.md)**: Complete installation and configuration guide

### Examples

- **[example_benchbase_integration.py](example_benchbase_integration.py)**: Comprehensive usage examples

## Quick Start

### 1. Install BenchBase

```bash
git clone --depth 1 https://github.com/cmu-db/benchbase.git
cd benchbase
./mvnw clean package -P postgres
cd target && tar xvzf benchbase-postgres.tgz
```

### 2. Setup Database

```bash
sudo -u postgres psql <<EOF
CREATE DATABASE benchbase;
CREATE USER benchuser WITH PASSWORD 'benchpass';
GRANT ALL PRIVILEGES ON DATABASE benchbase TO benchuser;
EOF
```

### 3. Install Python Dependencies

```bash
pip install psycopg2-binary numpy torch botorch
```

### 4. Run Example

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import torch

func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS
    db_host="localhost",
    db_port=5432,
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass"
)

# Evaluate a configuration
X = torch.rand(1, func.dim)
performance = func(X)
print(f"Performance: {performance.item():.2f} ms")
```

## Examples Overview

### Example 1: Direct Wrapper Usage

Shows how to use the BenchBaseWrapper class directly to run benchmarks and apply configurations.

### Example 2: DatabaseTuningFunction

Demonstrates using the BOMegaBench DatabaseTuningFunction interface with BenchBase.

### Example 3: Bayesian Optimization

Complete example of optimizing database configuration using Bayesian Optimization.

### Example 4: Custom Knobs

Shows how to define and tune custom database knob configurations.

### Example 5: Database Setup

One-time setup script for initializing benchmark data.

## Supported Workloads

- **tpcc**: TPC-C (OLTP benchmark)
- **tpch**: TPC-H (OLAP benchmark)
- **ycsb**: Yahoo! Cloud Serving Benchmark
- **tatp**: Telecom Application Transaction Processing
- **epinions**: Social network review site
- **seats**: Airline ticket reservation
- **smallbank**: Banking transactions
- And 11+ more...

## Supported Databases

- PostgreSQL 12+
- MySQL 8.0+
- MariaDB 10.5+
- SQLite
- CockroachDB

## Documentation

For detailed information, see:

- [BenchBase Integration Report](../../BENCHBASE_INTEGRATION_REPORT.md): Complete integration documentation
- [Database Tuning Guide](../../DATABASE_TUNING_INTEGRATION_GUIDE.md): Original integration guide
- [BenchBase Repository](https://github.com/cmu-db/benchbase): Official BenchBase documentation

## Common Issues

### Java not found
```bash
sudo apt-get install openjdk-11-jdk
```

### Database connection failed
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

### psycopg2 installation error
```bash
sudo apt-get install libpq-dev python3-dev
pip install psycopg2-binary
```

## Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the existing example structure
3. Add documentation at the top of the file
4. Update this README with a brief description

## Support

- BenchBase issues: https://github.com/cmu-db/benchbase/issues
- BOMegaBench issues: Open an issue in the main repository

## License

This code is part of BOMegaBench and follows the same license.
BenchBase is licensed under Apache 2.0.
