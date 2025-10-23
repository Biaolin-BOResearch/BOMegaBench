"""
BenchBase Wrapper for Database Benchmarking.

This module provides a Python wrapper around BenchBase, a multi-DBMS SQL
benchmarking framework via JDBC. BenchBase is a multi-threaded load generator
that produces variable rate, variable mixture load against any JDBC-enabled
relational database.

BenchBase is the official modernized version of the original OLTPBench.
Repository: https://github.com/cmu-db/benchbase
"""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import csv
import time
from datetime import datetime
import warnings


class BenchBaseWrapper:
    """
    Python wrapper for BenchBase database benchmarking tool.

    This class provides methods to:
    - Configure database connections
    - Apply database configuration knobs
    - Run benchmarks (TPC-C, TPC-H, YCSB, etc.)
    - Parse and return performance metrics

    Parameters
    ----------
    benchbase_path : str
        Path to the BenchBase installation directory (containing benchbase.jar)
    database_type : str
        Database type (e.g., "postgres", "mysql", "mariadb", "sqlite", "cockroachdb")
    db_host : str, optional
        Database host address (default: "localhost")
    db_port : int, optional
        Database port (default: 5432 for PostgreSQL)
    db_name : str, optional
        Database name (default: "benchbase")
    db_user : str, optional
        Database username
    db_password : str, optional
        Database password

    Examples
    --------
    >>> wrapper = BenchBaseWrapper(
    ...     benchbase_path="/path/to/benchbase-postgres",
    ...     database_type="postgres",
    ...     db_host="localhost",
    ...     db_port=5432,
    ...     db_name="testdb",
    ...     db_user="postgres",
    ...     db_password="password"
    ... )
    >>>
    >>> # Apply database knobs
    >>> knobs = {
    ...     "shared_buffers": "1GB",
    ...     "work_mem": "16MB",
    ...     "max_connections": "200"
    ... }
    >>> wrapper.apply_database_config(knobs)
    >>>
    >>> # Run TPC-C benchmark
    >>> result = wrapper.run_benchmark(
    ...     benchmark="tpcc",
    ...     warehouses=1,
    ...     terminals=1,
    ...     runtime_seconds=60
    ... )
    >>> print(f"Throughput: {result['throughput_txns_sec']}")
    >>> print(f"Avg Latency: {result['avg_latency_ms']}")
    """

    # Supported benchmarks in BenchBase
    SUPPORTED_BENCHMARKS = [
        "tpcc", "tpch", "tatp", "wikipedia", "resourcestresser",
        "twitter", "epinions", "ycsb", "seats", "auctionmark",
        "chbenchmark", "voter", "sibench", "noop", "smallbank",
        "hyadapt", "otmetrics", "templated"
    ]

    # Default database ports
    DEFAULT_PORTS = {
        "postgres": 5432,
        "mysql": 3306,
        "mariadb": 3306,
        "sqlite": None,
        "cockroachdb": 26257,
    }

    # JDBC driver classes
    JDBC_DRIVERS = {
        "postgres": "org.postgresql.Driver",
        "mysql": "com.mysql.cj.jdbc.Driver",
        "mariadb": "org.mariadb.jdbc.Driver",
        "sqlite": "org.sqlite.JDBC",
        "cockroachdb": "org.postgresql.Driver",
    }

    def __init__(
        self,
        benchbase_path: str,
        database_type: str = "postgres",
        db_host: str = "localhost",
        db_port: Optional[int] = None,
        db_name: str = "benchbase",
        db_user: str = "postgres",
        db_password: str = "password",
        results_dir: Optional[str] = None
    ):
        """Initialize BenchBase wrapper."""
        self.benchbase_path = Path(benchbase_path)
        self.database_type = database_type.lower()
        self.db_host = db_host
        self.db_port = db_port or self.DEFAULT_PORTS.get(self.database_type, 5432)
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password

        # Results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self.benchbase_path / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Validate BenchBase installation
        self._validate_installation()

        # Database connection handle (for applying knobs)
        self.db_connection = None

    def _validate_installation(self):
        """Validate that BenchBase is properly installed."""
        jar_path = self.benchbase_path / "benchbase.jar"
        if not jar_path.exists():
            raise FileNotFoundError(
                f"benchbase.jar not found at {jar_path}\n"
                f"Please install BenchBase following the instructions in the documentation."
            )

        # Check if Java is available
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Java is not installed or not in PATH. BenchBase requires Java to run."
            )

    def _get_jdbc_url(self) -> str:
        """Generate JDBC URL for the database connection."""
        if self.database_type == "postgres":
            return (
                f"jdbc:postgresql://{self.db_host}:{self.db_port}/{self.db_name}"
                f"?preferQueryMode=simple&sslmode=disable"
            )
        elif self.database_type in ["mysql", "mariadb"]:
            return (
                f"jdbc:mysql://{self.db_host}:{self.db_port}/{self.db_name}"
                f"?useSSL=false&allowPublicKeyRetrieval=true"
            )
        elif self.database_type == "sqlite":
            return f"jdbc:sqlite:{self.db_name}.db"
        elif self.database_type == "cockroachdb":
            return (
                f"jdbc:postgresql://{self.db_host}:{self.db_port}/{self.db_name}"
                f"?sslmode=disable"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _create_config_xml(
        self,
        benchmark: str,
        scale_factor: int = 1,
        terminals: int = 1,
        runtime_seconds: int = 60,
        rate: int = 10000,
        transaction_weights: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Create BenchBase XML configuration file.

        Parameters
        ----------
        benchmark : str
            Benchmark name (e.g., "tpcc", "ycsb")
        scale_factor : int
            Scale factor (e.g., number of warehouses for TPC-C)
        terminals : int
            Number of terminals/clients
        runtime_seconds : int
            Benchmark runtime in seconds
        rate : int
            Target transaction rate
        transaction_weights : str, optional
            Transaction weights (e.g., "45,43,4,4,4" for TPC-C)

        Returns
        -------
        Path
            Path to the generated configuration file
        """
        # Create root element
        root = ET.Element("parameters")

        # Database type
        ET.SubElement(root, "type").text = self.database_type.upper()

        # JDBC driver
        ET.SubElement(root, "driver").text = self.JDBC_DRIVERS[self.database_type]

        # JDBC URL
        ET.SubElement(root, "url").text = self._get_jdbc_url()

        # Database credentials
        ET.SubElement(root, "username").text = self.db_user
        ET.SubElement(root, "password").text = self.db_password

        # Isolation level
        ET.SubElement(root, "isolation").text = "TRANSACTION_SERIALIZABLE"

        # Batch size
        ET.SubElement(root, "batchsize").text = "128"

        # Scale factor
        ET.SubElement(root, "scalefactor").text = str(scale_factor)

        # Workload configuration
        works = ET.SubElement(root, "works")
        work = ET.SubElement(works, "work")

        ET.SubElement(work, "time").text = str(runtime_seconds)
        ET.SubElement(work, "rate").text = str(rate)

        if transaction_weights:
            ET.SubElement(work, "weights").text = transaction_weights

        # Terminals
        ET.SubElement(root, "terminals").text = str(terminals)

        # Create XML tree
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")

        # Write to temporary file
        config_file = self.results_dir / f"{benchmark}_config_{int(time.time())}.xml"
        tree.write(config_file, encoding="utf-8", xml_declaration=True)

        return config_file

    def apply_database_config(self, knobs: Dict[str, Any]) -> bool:
        """
        Apply database configuration knobs.

        This method applies configuration settings to the database.
        For PostgreSQL, this modifies postgresql.conf and reloads/restarts.
        For MySQL, this modifies my.cnf and restarts.

        Parameters
        ----------
        knobs : Dict[str, Any]
            Dictionary of knob names to values

        Returns
        -------
        bool
            True if configuration was successfully applied

        Notes
        -----
        This method requires appropriate permissions to modify database
        configuration and restart the database service.

        For safety, this implementation uses SQL ALTER SYSTEM commands
        when possible (PostgreSQL) or generates configuration snippets
        that should be manually applied.
        """
        if self.database_type == "postgres":
            return self._apply_postgres_config(knobs)
        elif self.database_type in ["mysql", "mariadb"]:
            return self._apply_mysql_config(knobs)
        else:
            warnings.warn(
                f"Configuration application not implemented for {self.database_type}"
            )
            return False

    def _apply_postgres_config(self, knobs: Dict[str, Any]) -> bool:
        """Apply PostgreSQL configuration knobs."""
        try:
            import psycopg2

            # Connect to database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Apply each knob using ALTER SYSTEM
            for knob_name, value in knobs.items():
                # Convert knob name format (e.g., shared_buffers_mb -> shared_buffers)
                pg_knob_name = knob_name.replace("_mb", "").replace("_gb", "")

                # Format value
                if knob_name.endswith("_mb"):
                    pg_value = f"{value}MB"
                elif knob_name.endswith("_gb"):
                    pg_value = f"{value}GB"
                else:
                    pg_value = str(value)

                try:
                    cursor.execute(f"ALTER SYSTEM SET {pg_knob_name} = %s", (pg_value,))
                except Exception as e:
                    warnings.warn(f"Could not set {pg_knob_name}: {e}")

            # Reload configuration
            cursor.execute("SELECT pg_reload_conf()")

            cursor.close()
            conn.close()

            # Wait for configuration to take effect
            time.sleep(2)

            return True

        except ImportError:
            warnings.warn(
                "psycopg2 not installed. Cannot apply PostgreSQL configuration.\n"
                "Install it with: pip install psycopg2-binary"
            )
            return False
        except Exception as e:
            warnings.warn(f"Error applying PostgreSQL configuration: {e}")
            return False

    def _apply_mysql_config(self, knobs: Dict[str, Any]) -> bool:
        """Apply MySQL configuration knobs."""
        try:
            import mysql.connector

            # Connect to database
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()

            # Apply each knob using SET GLOBAL
            for knob_name, value in knobs.items():
                # Convert knob name format
                mysql_knob_name = knob_name.replace("_mb", "").replace("_gb", "")

                # Format value
                if knob_name.endswith("_mb"):
                    mysql_value = value * 1024 * 1024  # Convert to bytes
                elif knob_name.endswith("_gb"):
                    mysql_value = value * 1024 * 1024 * 1024  # Convert to bytes
                else:
                    mysql_value = value

                try:
                    cursor.execute(f"SET GLOBAL {mysql_knob_name} = %s", (mysql_value,))
                except Exception as e:
                    warnings.warn(f"Could not set {mysql_knob_name}: {e}")

            conn.commit()
            cursor.close()
            conn.close()

            # Wait for configuration to take effect
            time.sleep(2)

            return True

        except ImportError:
            warnings.warn(
                "mysql-connector-python not installed. Cannot apply MySQL configuration.\n"
                "Install it with: pip install mysql-connector-python"
            )
            return False
        except Exception as e:
            warnings.warn(f"Error applying MySQL configuration: {e}")
            return False

    def run_benchmark(
        self,
        benchmark: str,
        create: bool = False,
        load: bool = False,
        execute: bool = True,
        scale_factor: int = 1,
        terminals: int = 1,
        runtime_seconds: int = 60,
        rate: int = 10000,
        transaction_weights: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a BenchBase benchmark.

        Parameters
        ----------
        benchmark : str
            Benchmark name (e.g., "tpcc", "ycsb", "tpch")
        create : bool
            Initialize the database schema (default: False)
        load : bool
            Load benchmark data (default: False)
        execute : bool
            Execute the benchmark workload (default: True)
        scale_factor : int
            Scale factor (e.g., warehouses for TPC-C) (default: 1)
        terminals : int
            Number of concurrent terminals/clients (default: 1)
        runtime_seconds : int
            Benchmark duration in seconds (default: 60)
        rate : int
            Target transaction rate (default: 10000)
        transaction_weights : str, optional
            Transaction type weights (e.g., "45,43,4,4,4" for TPC-C)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance metrics:
            - throughput_txns_sec: Transactions per second
            - avg_latency_ms: Average latency in milliseconds
            - p50_latency_ms: 50th percentile latency
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - total_transactions: Total number of transactions
            - success: Whether benchmark completed successfully

        Examples
        --------
        >>> # First time: create schema and load data
        >>> result = wrapper.run_benchmark(
        ...     benchmark="tpcc",
        ...     create=True,
        ...     load=True,
        ...     execute=True,
        ...     scale_factor=1,
        ...     terminals=4,
        ...     runtime_seconds=60
        ... )
        >>>
        >>> # Subsequent runs: just execute
        >>> result = wrapper.run_benchmark(
        ...     benchmark="tpcc",
        ...     execute=True,
        ...     terminals=4,
        ...     runtime_seconds=60
        ... )
        """
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark: {benchmark}\n"
                f"Supported benchmarks: {', '.join(self.SUPPORTED_BENCHMARKS)}"
            )

        # Create configuration file
        config_file = self._create_config_xml(
            benchmark=benchmark,
            scale_factor=scale_factor,
            terminals=terminals,
            runtime_seconds=runtime_seconds,
            rate=rate,
            transaction_weights=transaction_weights,
            **kwargs
        )

        try:
            # Build command
            jar_path = self.benchbase_path / "benchbase.jar"
            cmd = [
                "java", "-jar", str(jar_path),
                "-b", benchmark,
                "-c", str(config_file),
                "-d", str(self.results_dir),
            ]

            if create:
                cmd.append("--create=true")
            if load:
                cmd.append("--load=true")
            if execute:
                cmd.append("--execute=true")

            # Run BenchBase
            print(f"Running BenchBase command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.benchbase_path),
                timeout=runtime_seconds + 300  # Add buffer time
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"BenchBase failed with exit code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            print("BenchBase output:")
            print(result.stdout)

            # Parse results
            if execute:
                metrics = self._parse_results(benchmark)
                metrics["success"] = True
                return metrics
            else:
                return {"success": True, "message": "Schema created/data loaded"}

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"BenchBase timed out after {runtime_seconds + 300} seconds")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "throughput_txns_sec": 0.0,
                "avg_latency_ms": float('inf')
            }
        finally:
            # Clean up config file
            if config_file.exists():
                config_file.unlink()

    def _parse_results(self, benchmark: str) -> Dict[str, Any]:
        """
        Parse BenchBase results from CSV files.

        BenchBase generates results in CSV format with columns like:
        - timestamp
        - transaction type
        - latency
        - etc.

        Parameters
        ----------
        benchmark : str
            Benchmark name

        Returns
        -------
        Dict[str, Any]
            Parsed performance metrics
        """
        # Find most recent results file
        pattern = f"{benchmark}_*.csv"
        result_files = sorted(
            self.results_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not result_files:
            warnings.warn(f"No result files found matching {pattern}")
            return {
                "throughput_txns_sec": 0.0,
                "avg_latency_ms": float('inf'),
                "p50_latency_ms": float('inf'),
                "p95_latency_ms": float('inf'),
                "p99_latency_ms": float('inf'),
                "total_transactions": 0
            }

        result_file = result_files[0]
        print(f"Parsing results from: {result_file}")

        # Parse CSV
        latencies = []
        transaction_count = 0

        try:
            with open(result_file, 'r') as f:
                # Try to parse as samples file
                reader = csv.DictReader(f)
                for row in reader:
                    # BenchBase CSV format varies, try common column names
                    latency = None
                    if 'Latency (microseconds)' in row:
                        latency = float(row['Latency (microseconds)']) / 1000.0  # Convert to ms
                    elif 'latency' in row:
                        latency = float(row['latency']) / 1000.0
                    elif 'Latency' in row:
                        latency = float(row['Latency']) / 1000.0

                    if latency is not None:
                        latencies.append(latency)
                        transaction_count += 1
        except Exception as e:
            warnings.warn(f"Error parsing results: {e}")

        # Calculate metrics
        if latencies:
            import numpy as np
            latencies_sorted = sorted(latencies)
            metrics = {
                "throughput_txns_sec": transaction_count / 60.0,  # Assume 60 second benchmark
                "avg_latency_ms": np.mean(latencies),
                "p50_latency_ms": np.percentile(latencies_sorted, 50),
                "p95_latency_ms": np.percentile(latencies_sorted, 95),
                "p99_latency_ms": np.percentile(latencies_sorted, 99),
                "total_transactions": transaction_count
            }
        else:
            # Try to extract summary from the file or use defaults
            metrics = {
                "throughput_txns_sec": 0.0,
                "avg_latency_ms": float('inf'),
                "p50_latency_ms": float('inf'),
                "p95_latency_ms": float('inf'),
                "p99_latency_ms": float('inf'),
                "total_transactions": 0
            }

        return metrics

    def cleanup(self):
        """Clean up resources."""
        if self.db_connection:
            try:
                self.db_connection.close()
            except:
                pass
