# BenchBase Integration Quick Start

Get started with BenchBase and BOMegaBench database tuning in 15 minutes.

## Prerequisites

- Java 11+ installed
- PostgreSQL running on localhost
- Python 3.8+ with pip

## Step 1: Install BenchBase (5 minutes)

```bash
# Clone and build
git clone --depth 1 https://github.com/cmu-db/benchbase.git
cd benchbase
./mvnw clean package -P postgres

# Extract
cd target
tar xvzf benchbase-postgres.tgz
cd benchbase-postgres

# Save this path - you'll need it later
pwd
# Example output: /home/user/benchbase/target/benchbase-postgres
```

## Step 2: Setup Database (3 minutes)

```bash
# Create database and user
sudo -u postgres psql <<EOF
CREATE DATABASE benchbase;
CREATE USER benchuser WITH PASSWORD 'benchpass';
GRANT ALL PRIVILEGES ON DATABASE benchbase TO benchuser;
EOF

# Test connection
psql -h localhost -U benchuser -d benchbase -c "SELECT 1;"
```

## Step 3: Install Python Dependencies (2 minutes)

```bash
pip install psycopg2-binary numpy torch botorch
```

## Step 4: Setup BenchBase Data (5 minutes)

```bash
# Go to BenchBase directory
cd /path/to/benchbase/target/benchbase-postgres

# Edit config file (or use sample)
# Update username/password if different
nano config/postgres/sample_tpcc_config.xml

# Create schema and load data (one time only)
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml \
    --create=true --load=true
```

## Step 5: Test Integration (2 minutes)

Create `test_quick.py`:

```python
import sys
sys.path.insert(0, '/path/to/BOMegaBench')  # Update this path

from bomegabench.functions.database_tuning import DatabaseTuningFunction
import torch

# Create function (update benchbase_path!)
func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    benchbase_path="/path/to/benchbase/target/benchbase-postgres",  # UPDATE THIS!
    db_host="localhost",
    db_port=5432,
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass",
    benchmark_runtime=30,  # Short test run
    benchmark_terminals=2
)

print(f"Testing {func.dim}-dimensional optimization space")

# Test evaluation
X = torch.rand(1, func.dim)
performance = func(X)
print(f"Performance: {performance.item():.2f} ms")
print("Integration working!")
```

Run:

```bash
python test_quick.py
```

## What's Next?

- See [BENCHBASE_SETUP.md](BENCHBASE_SETUP.md) for detailed setup
- See [example_benchbase_integration.py](example_benchbase_integration.py) for usage examples
- Try Bayesian Optimization (example 3 in examples file)
- Customize knobs for your workload

## Common Issues

**Java not found**: Install Java 11+
```bash
sudo apt-get install openjdk-11-jdk
```

**Database connection failed**: Check PostgreSQL is running
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

**Permission denied**: Grant database permissions
```bash
sudo -u postgres psql -d benchbase -c "GRANT ALL ON SCHEMA public TO benchuser;"
```

**psycopg2 error**: Install dev libraries
```bash
sudo apt-get install libpq-dev python3-dev
pip install psycopg2-binary
```

## Need Help?

- Full setup guide: [BENCHBASE_SETUP.md](BENCHBASE_SETUP.md)
- Examples: [example_benchbase_integration.py](example_benchbase_integration.py)
- BenchBase docs: https://github.com/cmu-db/benchbase
