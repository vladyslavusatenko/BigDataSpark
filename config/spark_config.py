"""
Spark Configuration for Black Friday Big Data Project
"""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os
import sys
import platform


class SparkConfig:
    """Spark configuration and session management"""

    @staticmethod
    def setup_windows_hadoop():
        """Setup Hadoop for Windows environment"""
        if platform.system() == 'Windows':
            # Use project root directory instead of current working directory
            # This ensures hadoop path is correct even when running from notebooks/
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            hadoop_dir = os.path.join(project_root, 'hadoop')
            bin_dir = os.path.join(hadoop_dir, 'bin')

            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir, exist_ok=True)

            # Set environment variables
            os.environ['HADOOP_HOME'] = hadoop_dir
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')

            print(f"✓ Windows: HADOOP_HOME set to {hadoop_dir}")
            print("  Note: If you encounter permission errors, download winutils.exe")
            print("  from https://github.com/steveloughran/winutils and place in hadoop/bin/")

    @staticmethod
    def get_spark_session(app_name="BlackFridayAnalysis", master="local[*]"):
        """
        Create and configure Spark session with Delta Lake support

        Args:
            app_name: Name of Spark application
            master: Spark master URL (local[*] for all cores)

        Returns:
            SparkSession configured with Delta Lake and optimizations
        """

        # Setup Windows environment if needed
        SparkConfig.setup_windows_hadoop()

        builder = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "100") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Windows-specific configurations
        if platform.system() == 'Windows':
            builder = builder \
                .config("spark.sql.warehouse.dir", "file:///C:/tmp/spark-warehouse") \
                .config("spark.driver.host", "127.0.0.1")

        # Configure Delta Lake
        spark = configure_spark_with_delta_pip(builder).getOrCreate()

        # Set log level
        spark.sparkContext.setLogLevel("WARN")

        print("=" * 60)
        print(f"Spark Session Created: {app_name}")
        print("=" * 60)
        print(f"Spark Version: {spark.version}")
        print(f"Master: {master}")
        print(f"App Name: {app_name}")
        print("=" * 60)

        return spark

    @staticmethod
    def get_streaming_session(app_name="BlackFridayStreaming"):
        """
        Create Spark session configured for streaming
        """

        spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        print(f"✓ Streaming Spark Session Created: {app_name}")

        return spark

    @staticmethod
    def stop_spark_session(spark):
        """Gracefully stop Spark session"""
        if spark:
            spark.stop()
            print("✓ Spark session stopped")


# Spark SQL optimization configurations
SPARK_CONFIGS = {
    # Memory Management
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.3",

    # Shuffle Optimization
    "spark.sql.shuffle.partitions": "200",
    "spark.shuffle.file.buffer": "64k",
    "spark.reducer.maxSizeInFlight": "96m",

    # Adaptive Query Execution
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",

    # Performance
    "spark.default.parallelism": "100",
    "spark.sql.files.maxPartitionBytes": "128MB",
    "spark.sql.autoBroadcastJoinThreshold": "10MB",

    # Delta Lake
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",

    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryo.registrationRequired": "false"
}

# Kafka configurations for streaming
KAFKA_CONFIGS = {
    "kafka.bootstrap.servers": "localhost:9092",
    "subscribe": "black-friday-purchases",
    "startingOffsets": "earliest",
    "failOnDataLoss": "false",
    "kafka.group.id": "black-friday-consumer",
    "maxOffsetsPerTrigger": "1000"
}


# Data paths
class DataPaths:
    """Project data paths"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    RAW_DATA = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA = os.path.join(DATA_DIR, "processed")
    STREAMING_DATA = os.path.join(DATA_DIR, "streaming")

    # Delta Lake paths
    DELTA_PURCHASES = os.path.join(PROCESSED_DATA, "delta", "purchases")
    DELTA_CUSTOMERS = os.path.join(PROCESSED_DATA, "delta", "customers")
    DELTA_PRODUCTS = os.path.join(PROCESSED_DATA, "delta", "products")

    # Model paths
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    @staticmethod
    def create_directories():
        """Create all necessary directories"""
        dirs = [
            DataPaths.RAW_DATA,
            DataPaths.PROCESSED_DATA,
            DataPaths.STREAMING_DATA,
            DataPaths.MODELS_DIR
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        print("✓ Project directories created")


if __name__ == "__main__":
    # Test configuration
    print("\nTesting Spark Configuration...\n")

    # Create directories
    DataPaths.create_directories()

    # Create Spark session
    spark = SparkConfig.get_spark_session()

    # Test with simple DataFrame
    data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    df = spark.createDataFrame(data, ["name", "age"])

    print("\nTest DataFrame:")
    df.show()

    # Stop session
    SparkConfig.stop_spark_session(spark)

    print("\n✅ Configuration test completed!")