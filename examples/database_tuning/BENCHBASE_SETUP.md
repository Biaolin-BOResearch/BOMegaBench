# BenchBase Setup Guide for BOMegaBench

This guide provides step-by-step instructions for installing and configuring BenchBase for use with BOMegaBench's database tuning module.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installing BenchBase](#installing-benchbase)
4. [Database Setup](#database-setup)
5. [Python Dependencies](#python-dependencies)
6. [Verifying Installation](#verifying-installation)
7. [Troubleshooting](#troubleshooting)

## Overview

**BenchBase** is a Multi-DBMS SQL Benchmarking Framework via JDBC, developed by Carnegie Mellon University's Database Group. It is the official modernized version of the original OLTPBench.

- **Repository**: https://github.com/cmu-db/benchbase
- **Supported Databases**: PostgreSQL, MySQL, MariaDB, SQLite, CockroachDB, and more
- **Supported Benchmarks**: TPC-C, TPC-H, YCSB, TATP, and 15+ others

## Prerequisites

### System Requirements

- **Java**: JDK 11 or higher
- **Maven**: 3.6 or higher (bundled with BenchBase using Maven wrapper)
- **Python**: 3.8 or higher
- **Database**: PostgreSQL 12+ or MySQL 8.0+ (or other supported DBMS)
- **Memory**: At least 4GB RAM (8GB+ recommended)
- **Disk Space**: 2GB for BenchBase + space for database

### Check Java Installation

```bash
java -version
```

Expected output: `java version "11.0.x"` or higher

If Java is not installed:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# macOS (using Homebrew)
brew install openjdk@11

# Windows
# Download from: https://adoptium.net/
```

## Installing BenchBase

### Step 1: Clone the Repository

```bash
cd /path/to/your/workspace
git clone --depth 1 https://github.com/cmu-db/benchbase.git
cd benchbase
```

### Step 2: Build for Your Database

Build BenchBase with the appropriate database profile:

#### For PostgreSQL:

```bash
./mvnw clean package -P postgres
```

#### For MySQL:

```bash
./mvnw clean package -P mysql
```

#### For MariaDB:

```bash
./mvnw clean package -P mariadb
```

#### For Multiple Databases:

```bash
./mvnw clean package -P postgres,mysql
```

**Note**: The first build may take 5-10 minutes as Maven downloads dependencies.

### Step 3: Extract the Build Artifact

```bash
cd target
tar xvzf benchbase-postgres.tgz
cd benchbase-postgres
```

Your BenchBase installation is now at: `/path/to/your/workspace/benchbase/target/benchbase-postgres`

### Step 4: Verify Installation

```bash
java -jar benchbase.jar -h
```

You should see the BenchBase help message with available options.

## Database Setup

### PostgreSQL Setup

#### 1. Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# macOS (using Homebrew)
brew install postgresql@14
brew services start postgresql@14

# Windows
# Download installer from: https://www.postgresql.org/download/windows/
```

#### 2. Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE benchbase;
CREATE USER benchuser WITH PASSWORD 'benchpass';
GRANT ALL PRIVILEGES ON DATABASE benchbase TO benchuser;
\q
```

#### 3. Configure PostgreSQL for BenchBase

Edit `postgresql.conf` (usually in `/etc/postgresql/14/main/postgresql.conf`):

```conf
# Connection settings
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 4GB
work_mem = 16MB

# Performance settings (adjust based on your hardware)
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

#### 4. Test Connection

```bash
psql -h localhost -U benchuser -d benchbase -c "SELECT version();"
```

### MySQL Setup

#### 1. Install MySQL

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mysql-server

# macOS (using Homebrew)
brew install mysql
brew services start mysql

# Windows
# Download installer from: https://dev.mysql.com/downloads/installer/
```

#### 2. Create Database and User

```bash
sudo mysql

# In MySQL shell:
CREATE DATABASE benchbase;
CREATE USER 'benchuser'@'localhost' IDENTIFIED BY 'benchpass';
GRANT ALL PRIVILEGES ON benchbase.* TO 'benchuser'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### 3. Configure MySQL for BenchBase

Edit `my.cnf` (usually in `/etc/mysql/my.cnf`):

```conf
[mysqld]
innodb_buffer_pool_size = 2G
innodb_log_file_size = 512M
max_connections = 200
innodb_flush_log_at_trx_commit = 2
innodb_flush_method = O_DIRECT
```

Restart MySQL:

```bash
sudo systemctl restart mysql
```

#### 4. Test Connection

```bash
mysql -h localhost -u benchuser -pbenchpass benchbase -e "SELECT VERSION();"
```

## Python Dependencies

### Install Required Packages

```bash
# For PostgreSQL
pip install psycopg2-binary

# For MySQL
pip install mysql-connector-python

# For BOMegaBench (if not already installed)
pip install torch numpy

# Optional: For Bayesian Optimization
pip install botorch
```

### Create Requirements File

Create `requirements-database-tuning.txt`:

```
psycopg2-binary>=2.9.0
mysql-connector-python>=8.0.0
numpy>=1.21.0
torch>=1.10.0
botorch>=0.6.0  # Optional, for optimization
```

Install all at once:

```bash
pip install -r requirements-database-tuning.txt
```

## Verifying Installation

### Test BenchBase Directly

#### 1. Create Sample Configuration

BenchBase includes sample configurations in the `config/` directory. For PostgreSQL TPC-C:

```bash
cd /path/to/benchbase/target/benchbase-postgres
ls config/postgres/
```

You should see files like `sample_tpcc_config.xml`.

#### 2. Edit Configuration

Edit `config/postgres/sample_tpcc_config.xml` to match your database settings:

```xml
<?xml version="1.0"?>
<parameters>
    <type>POSTGRES</type>
    <driver>org.postgresql.Driver</driver>
    <url>jdbc:postgresql://localhost:5432/benchbase?preferQueryMode=simple&amp;sslmode=disable</url>
    <username>benchuser</username>
    <password>benchpass</password>
    <isolation>TRANSACTION_SERIALIZABLE</isolation>
    <batchsize>128</batchsize>
    <scalefactor>1</scalefactor>
    <terminals>1</terminals>
    <works>
        <work>
            <time>60</time>
            <rate>10000</rate>
            <weights>45,43,4,4,4</weights>
        </work>
    </works>
</parameters>
```

#### 3. Run Test Benchmark

```bash
# Create schema
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create=true

# Load data
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --load=true

# Run benchmark
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --execute=true
```

#### 4. Check Results

Results are saved in the `results/` directory:

```bash
ls -lh results/
cat results/tpcc_*.summary.json  # Summary statistics
```

### Test Python Integration

Create `test_integration.py`:

```python
from bomegabench.functions.benchbase_wrapper import BenchBaseWrapper

# Update this path
BENCHBASE_PATH = "/path/to/benchbase/target/benchbase-postgres"

wrapper = BenchBaseWrapper(
    benchbase_path=BENCHBASE_PATH,
    database_type="postgres",
    db_host="localhost",
    db_port=5432,
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass"
)

print("BenchBase wrapper initialized successfully!")

# Test configuration application
config = {
    "shared_buffers_mb": 1024,
    "work_mem_mb": 16
}

print("Applying configuration:", config)
result = wrapper.apply_database_config(config)
print("Configuration applied:", result)
```

Run the test:

```bash
python test_integration.py
```

## Troubleshooting

### Common Issues

#### 1. Java Not Found

**Error**: `java: command not found`

**Solution**:
```bash
# Verify Java installation
which java
java -version

# Add to PATH if needed (add to ~/.bashrc or ~/.zshrc)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

#### 2. Maven Build Fails

**Error**: `[ERROR] Failed to execute goal`

**Solution**:
```bash
# Clean and rebuild
./mvnw clean
./mvnw package -P postgres -DskipTests
```

#### 3. Database Connection Refused

**Error**: `Connection refused` or `Cannot connect to database`

**Solution**:
```bash
# Check if database is running
sudo systemctl status postgresql
# or
sudo systemctl status mysql

# Check connection manually
psql -h localhost -U benchuser -d benchbase
# or
mysql -h localhost -u benchuser -pbenchpass benchbase

# Check firewall settings
sudo ufw status
```

#### 4. Permission Denied on PostgreSQL

**Error**: `permission denied for database benchbase`

**Solution**:
```sql
-- Connect as postgres user
sudo -u postgres psql

-- Grant all permissions
GRANT ALL PRIVILEGES ON DATABASE benchbase TO benchuser;
ALTER DATABASE benchbase OWNER TO benchuser;

-- For PostgreSQL 15+, also grant schema permissions
\c benchbase
GRANT ALL ON SCHEMA public TO benchuser;
```

#### 5. Out of Memory

**Error**: `Java heap space` or `OutOfMemoryError`

**Solution**:
```bash
# Increase Java heap size
export MAVEN_OPTS="-Xmx4g"
./mvnw clean package -P postgres

# Or when running BenchBase
java -Xmx2g -jar benchbase.jar -b tpcc -c config.xml --execute=true
```

#### 6. Results Not Found

**Error**: `No result files found matching tpcc_*.csv`

**Solution**:
- Ensure `--execute=true` was used when running the benchmark
- Check the results directory: `ls -la results/`
- Look for `.summary.json` or `.samples.csv` files
- Verify the benchmark completed successfully (check stdout/stderr)

#### 7. psycopg2 Installation Fails

**Error**: `Error: pg_config executable not found`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libpq-dev python3-dev

# macOS
brew install postgresql

# Then reinstall
pip install psycopg2-binary
```

### Getting Help

- **BenchBase Issues**: https://github.com/cmu-db/benchbase/issues
- **BenchBase Wiki**: https://github.com/cmu-db/benchbase/wiki
- **BOMegaBench Issues**: https://github.com/your-org/BOMegaBench/issues

## Next Steps

After completing the setup:

1. Review the [example_benchbase_integration.py](example_benchbase_integration.py) for usage examples
2. Read the [main documentation](../../DATABASE_TUNING_INTEGRATION_GUIDE.md) for integration details
3. Try running a simple optimization experiment
4. Customize knob configurations for your use case

## Quick Reference

### Directory Structure

```
/path/to/benchbase/
├── target/
│   └── benchbase-postgres/
│       ├── benchbase.jar          # Main executable
│       ├── config/                # Configuration templates
│       │   └── postgres/
│       │       └── sample_tpcc_config.xml
│       ├── results/               # Benchmark results
│       └── lib/                   # JDBC drivers
```

### Common Commands

```bash
# Build BenchBase
./mvnw clean package -P postgres

# Create schema
java -jar benchbase.jar -b tpcc -c config.xml --create=true

# Load data
java -jar benchbase.jar -b tpcc -c config.xml --load=true

# Run benchmark
java -jar benchbase.jar -b tpcc -c config.xml --execute=true

# All in one
java -jar benchbase.jar -b tpcc -c config.xml --create=true --load=true --execute=true
```

### Supported Benchmarks

| Benchmark | Description | Scale Parameter |
|-----------|-------------|-----------------|
| tpcc | TPC-C (OLTP) | Warehouses |
| tpch | TPC-H (OLAP) | Scale factor |
| ycsb | Yahoo! Cloud Serving Benchmark | Record count |
| tatp | Telecom Application Transaction Processing | Subscribers |
| epinions | Social network review site | Users |
| seats | Airline ticket reservation | Flights |
| smallbank | Banking transactions | Customers |

## Additional Resources

- [BenchBase GitHub](https://github.com/cmu-db/benchbase)
- [BenchBase Documentation](https://illuminatedcomputing.com/posts/2024/08/benchbase-documentation/)
- [TPC-C Specification](http://www.tpc.org/tpcc/)
- [PostgreSQL Tuning](https://wiki.postgresql.org/wiki/Tuning_Your_PostgreSQL_Server)
- [MySQL Tuning](https://dev.mysql.com/doc/refman/8.0/en/optimization.html)
