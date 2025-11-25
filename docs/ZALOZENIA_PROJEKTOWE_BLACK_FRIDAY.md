# PROJEKT BIG DATA - BLACK FRIDAY SALES ANALYSIS
## Założenia Projektowe i Dokumentacja


## 1. OPIS CELU PROJEKTU

### 1.1. Temat Projektu
**"Analiza i Predykcja Zachowań Zakupowych Klientów podczas Black Friday z wykorzystaniem Apache Spark"**

### 1.2. Cel Biznesowy
Projekt ma na celu stworzenie kompleksowego systemu analitycznego do:
- Prognozowania wartości zakupów klientów
- Segmentacji klientów na podstawie zachowań zakupowych
- Budowy systemu rekomendacji produktów
- Identyfikacji wzorców zakupowych i trendów

### 1.3. Tezy Badawcze

#### Teza 1: Predykcja Wartości Zakupu
**Hipoteza:** Na podstawie cech demograficznych klienta (wiek, płeć, stan cywilny, miasto) oraz historii zakupów można przewidzieć wartość przyszłych transakcji z dokładnością >85%.

**Metody weryfikacji:**
- Regresja liniowa (Spark MLlib)
- Random Forest Regressor
- Gradient Boosted Trees
- Metryki: RMSE, MAE, R²

#### Teza 2: Segmentacja Klientów
**Hipoteza:** Istnieją wyraźne segmenty klientów o odmiennych wzorcach zakupowych, które można zidentyfikować za pomocą metod klastrowania.

**Metody weryfikacji:**
- K-means clustering (Spark MLlib)
- Analiza skupień według cech: wartość zakupu, częstotliwość, kategorie produktów
- Metryki: Silhouette Score, Within-Cluster Sum of Squares

#### Teza 3: System Rekomendacji
**Hipoteza:** Model collaborative filtering może skutecznie rekomendować produkty na podstawie zachowań podobnych użytkowników.

**Metody weryfikacji:**
- ALS (Alternating Least Squares) algorithm
- Matrix factorization
- Metryki: Precision@K, Recall@K, NDCG

#### Teza 4: Wzorce Demograficzne
**Hipoteza:** Istnieją statystycznie istotne różnice w zachowaniach zakupowych między różnymi grupami demograficznymi.

**Metody weryfikacji:**
- Analiza wariancji (ANOVA)
- Chi-kwadrat dla zmiennych kategorycznych
- Testy post-hoc (Tukey HSD)

### 1.4. Założenia Narzędziowe i Metodologia

#### Apache Spark - Rdzeń Platformy Big Data
**Uzasadnienie wyboru:**
- Możliwość przetwarzania danych o wielkości przekraczającej pamięć RAM
- Wbudowane API do uczenia maszynowego (MLlib)
- Natywne wsparcie dla streamingu danych
- Optymalizacja wykonania zapytań (Catalyst Optimizer)

**Wykorzystywane komponenty:**
- **Spark Core:** Przetwarzanie rozproszone, RDD transformations, distributed computing
- **Spark SQL:** Transformacje DataFrame, agregacje, window functions, złączenia tabel
- **Spark MLlib:** Regression (Linear, Random Forest, GBT), Clustering (K-Means), Collaborative Filtering (ALS), Feature transformations (VectorAssembler, StringIndexer, OneHotEncoder)
- **Spark Streaming:** Structured Streaming API, micro-batch processing, watermarking, stateful operations

#### Apache Kafka - Platforma Streamingowa
**Uzasadnienie wyboru:**
- Industry standard dla distributed streaming
- Wysoka przepustowość (miliony eventów/sec)
- Fault-tolerant i skalowalne architektura
- Naturalna integracja ze Spark Streaming

**Implementacja:**
- **Producer:** Symulator transakcji z kontrolowaną emisją eventów
- **Consumer:** Spark Structured Streaming z checkpointing
- **Topics:** `black-friday-purchases`, `customer-events`, `product-views`
- **Retention:** 7 dni (time-based retention policy)

#### Delta Lake - Lakehouse Architecture
**Uzasadnienie wyboru:**
- ACID transactions na data lake
- Time travel i versioning danych
- Schema evolution i enforcement
- Optymalizacja dla Spark (Z-Ordering, Data Skipping)

**Zastosowanie:**
- Przechowywanie processed data
- Incremental updates ze streaming
- Audit trail wszystkich zmian
- Rollback w przypadku błędów

#### Platforma Wykonawcza
**Wybór 1: Lokalne środowisko (deweloperskie)**
- Windows 11 + WSL2 / Ubuntu
- Python 3.11 + PySpark 3.5
- Jupyter Notebook / VS Code
- Docker dla Kafka

**Wybór 2: Google Colab (prototyping)**
- Darmowe GPU/TPU dla ML
- Pre-instalowany PySpark
- Łatwe udostępnianie notebooks

**Wybór 3: Azure Databricks (produkcyjne)**
- Managed Spark clusters
- Collaborative workspace
- MLflow integration
- Production-ready deployment

#### Stack Technologiczny - Pełna Lista
```
Data Processing:
├── Apache Spark 3.5.x (PySpark)
├── Apache Kafka 3.6.x
├── Delta Lake 3.0.x
└── Hadoop 3.3.x (Windows compatibility)

Machine Learning:
├── Spark MLlib (distributed ML)
├── Scikit-learn (baseline models)
└── MLflow (experiment tracking)

Data Analysis & Visualization:
├── Pandas 2.1.x
├── NumPy 1.26.x
├── Matplotlib 3.8.x
├── Seaborn 0.13.x
└── Plotly 5.18.x (interactive dashboards)

Development Tools:
├── Jupyter Notebook / JupyterLab
├── Git (version control)
├── Poetry / pip (dependency management)
└── pytest (testing)
```

### 1.5. Metody Statystyczne i Eksploracyjne

#### Analiza Eksploracyjna (EDA)
- Statystyki opisowe (średnia, mediana, odchylenie standardowe)
- Rozkłady zmiennych (histogramy, box plots)
- Analiza korelacji między zmiennymi
- Wykrywanie wartości odstających (outliers)
- Analiza brakujących danych

#### Metody Statystyczne
- Testy normalności rozkładu (Shapiro-Wilk, Kolmogorov-Smirnov)
- Testy istotności różnic (t-test, ANOVA)
- Analiza korelacji (Pearson, Spearman)
- Testy niezależności (Chi-kwadrat)

#### Feature Engineering
- One-hot encoding dla zmiennych kategorycznych
- Normalizacja/standaryzacja zmiennych numerycznych
- Tworzenie cech agregacyjnych (total_spend, purchase_frequency)
- Binning dla zmiennych ciągłych

### 1.6. Metodologia ETL i Transformacji Danych

#### Extract (Ekstrakcja)
**Źródła danych:**
1. **Batch Source:** CSV files → Spark DataFrame
2. **Streaming Source:** Kafka topics → Structured Streaming DataFrame

**Proces ekstrakcji:**
```python
# Batch ingestion
df_raw = spark.read.csv("data/raw/BlackFriday.csv", header=True, inferSchema=True)

# Streaming ingestion
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "black-friday-purchases") \
    .load()
```

#### Transform (Transformacja)
**Pipeline transformacji w Spark:**

**Krok 1: Data Quality & Cleansing**
- Usunięcie duplikatów (dropDuplicates)
- Handling missing values (imputation / removal)
- Outlier detection i treatment (IQR method, Z-score)
- Data type conversions

**Krok 2: Feature Engineering**
```python
# Agregacje per użytkownik
user_features = df.groupBy("User_ID").agg(
    count("*").alias("purchase_count"),
    sum("Purchase").alias("total_spent"),
    avg("Purchase").alias("avg_purchase"),
    min("Purchase").alias("min_purchase"),
    max("Purchase").alias("max_purchase"),
    countDistinct("Product_ID").alias("unique_products")
)

# RFM Analysis
rfm_features = calculate_rfm(df, reference_date)

# Temporal features (if timestamps available)
df = df.withColumn("day_of_week", dayofweek("timestamp"))
df = df.withColumn("hour_of_day", hour("timestamp"))
df = df.withColumn("is_weekend", when(col("day_of_week").isin([1,7]), 1).otherwise(0))
```

**Krok 3: Feature Encoding**
```python
# String Indexer dla zmiennych kategorycznych
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index")
            for col in categorical_cols]

# One-Hot Encoding
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_vec")
           for col in categorical_cols]

# Vector Assembler
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

# Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])
```

**Krok 4: Normalizacja**
```python
# StandardScaler dla zmiennych numerycznych
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# MinMaxScaler jako alternatywa
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
```

#### Load (Ładowanie)
**Formaty docelowe:**

**1. Delta Lake (Primary Storage)**
```python
# Batch write
df_transformed.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("City_Category", "Age") \
    .save("/delta/black-friday/processed")

# Streaming write
df_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/delta/checkpoints") \
    .start("/delta/black-friday/streaming")
```

**2. Parquet (Archival)**
```python
df.write.parquet("/data/processed/parquet/", mode="overwrite", compression="snappy")
```

**3. In-Memory (dla ML)**
```python
df.cache()  # Lazy caching
df.persist(StorageLevel.MEMORY_AND_DISK)  # Explicit persistence
```

#### Optymalizacja Pipeline ETL
**Spark optimizations:**
- **Partitioning:** Optimal partition size ~128MB
- **Broadcast joins:** dla małych tabel (<10MB)
- **Catalyst Optimizer:** wykorzystanie DataFrame API
- **Adaptive Query Execution (AQE):** dynamiczna optymalizacja
- **Z-Ordering:** dla Delta Lake queries
- **Data Skipping:** min/max statistics w Delta

**Monitoring i Quality Checks:**
```python
# Data quality assertions
assert df.count() > 0, "Empty DataFrame"
assert df.filter(col("Purchase").isNull()).count() == 0, "Null purchases found"
assert df.filter(col("Purchase") < 0).count() == 0, "Negative purchases found"

# Schema validation
expected_schema = StructType([...])
assert df.schema == expected_schema, "Schema mismatch"
```

---

## 2. DOBÓR DANYCH DO POSTAWIONEGO CELU

### 2.1. Charakterystyka Zbioru Danych

#### Źródło Danych
- **Pochodzenie:** Kaggle - Black Friday Sales Dataset
- **Link:** https://www.kaggle.com/datasets/sdolezel/black-friday
- **Licencja:** Open Database License (ODbL)
- **Rozmiar:** ~550,000 rekordów transakcyjnych

#### Natywna Struktura Danych
**Format:** CSV (Comma-Separated Values)

**Schemat danych (12 kolumn):**

| Kolumna | Typ | Opis | Przykład |
|---------|-----|------|----------|
| User_ID | Integer | Unikalny identyfikator klienta | 1000001 |
| Product_ID | String | Unikalny identyfikator produktu | P00069042 |
| Gender | String | Płeć klienta (M/F) | M |
| Age | String | Grupa wiekowa | 0-17, 18-25, 26-35, etc. |
| Occupation | Integer | Kod zawodu (zanonimizowany) | 0-20 |
| City_Category | String | Kategoria miasta | A, B, C |
| Stay_In_Current_City_Years | String | Lata zamieszkania | 0, 1, 2, 3, 4+ |
| Marital_Status | Integer | Stan cywilny (0=kawaler/panna, 1=żonaty/zamężna) | 0, 1 |
| Product_Category_1 | Integer | Główna kategoria produktu | 1-20 |
| Product_Category_2 | Integer | Druga kategoria produktu | 1-18 (nullable) |
| Product_Category_3 | Integer | Trzecia kategoria produktu | 1-18 (nullable) |
| Purchase | Integer | Wartość zakupu w USD | 185-23961 |

### 2.2. Charakter Ilościowy i Jakościowy Danych

#### Ilościowa Charakterystyka

**Rozmiar zbioru:**
- Liczba rekordów: ~550,000 transakcji
- Liczba unikalnych klientów: ~5,891
- Liczba unikalnych produktów: ~3,631
- Rozmiar pliku: ~57 MB (CSV)
- Średnia liczba transakcji na klienta: ~93

**Rozkład danych:**
- Zakres wartości zakupów: 185 - 23,961 USD
- Średnia wartość zakupu: ~9,263 USD
- Mediana wartości zakupu: ~8,062 USD

#### Jakościowa Charakterystyka

**Zmienne Kategoryczne:**
- Gender: 2 kategorie (Male ~75%, Female ~25%)
- Age: 7 grup wiekowych
- City_Category: 3 kategorie (A, B, C)
- Occupation: 21 kodów zawodów
- Stay_In_Current_City_Years: 5 kategorii

**Zmienne Numeryczne:**
- Purchase: zmienna ciągła (target variable)
- Product_Category_1/2/3: dyskretne wartości kategorii

**Jakość danych:**
- Product_Category_2: ~31% missing values
- Product_Category_3: ~69% missing values
- Pozostałe kolumny: brak wartości brakujących

### 2.3. Sposób Powstania Danych

#### Klasyfikacja Danych
**Dane wtórne** - zbiór został utworzony na podstawie:
- Historycznych danych transakcyjnych z systemów POS (Point of Sale)
- Danych z programów lojalnościowych klientów
- Zanonimizowanych danych e-commerce

#### Proces Zbierania Danych (hipotetyczny)
1. **Źródło pierwotne:** Systemy sprzedażowe retailera
2. **Anonimizacja:** Usunięcie danych osobowych (GDPR compliance)
3. **Agregacja:** Połączenie danych z różnych kanałów sprzedaży
4. **Eksport:** Konwersja do formatu CSV

### 2.4. Dynamika Zmian Danych

#### Typ Rejestracji
**Rejestracja w interwałach czasowych:**
- Każdy rekord reprezentuje pojedynczą transakcję zakupową
- Brak timestampów w datasecie (snapshot danych)
- Dane statyczne - jeden okres Black Friday

#### Charakterystyka Temporalna
- **Typ:** Dane przekrojowe (cross-sectional)
- **Okres:** Pojedyncze wydarzenie Black Friday
- **Częstotliwość:** Event-based (zdarzenie = transakcja)

### 2.5. Źródła Pochodzenia Danych

#### Źródło Bezpośrednie
- **Platforma:** Kaggle
- **Uploader:** @STEFANDOLEZEL
- **Publikacja:** 2018
- **Licencja:** Open Data Commons Open Database License (ODbL)

#### Źródło Pośrednie (założenia)
- Dane pochodzą z amerykańskiego retailera
- Prawdopodobnie agregacja z multiple stores
- Możliwe źródła: Walmart, Target, Amazon (nieoficjalne)

---

## 3. PROCES POZYSKIWANIA DANYCH - SYMULACJA STREAMING

### 3.1. Architektura Streaming Pipeline

Ponieważ oryginalny dataset jest statyczny, zaprojektowano **symulację streaming** dla celów projektu:

```
[Static CSV] → [Kafka Producer] → [Kafka Broker] → [Spark Streaming] → [Processing] → [Storage]
```

### 3.2. Apache Kafka - Konfiguracja

#### Topics
1. **black-friday-purchases**
   - Strumień transakcji zakupowych
   - Partycje: 3
   - Replication factor: 1 (dev environment)

2. **customer-events**
   - Wydarzenia związane z aktywnością klientów
   - Partycje: 2

#### Producer Configuration
```python
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'black-friday-producer',
    'acks': 'all',
    'compression.type': 'gzip'
}
```

#### Message Format (JSON)
```json
{
  "user_id": 1000001,
  "product_id": "P00069042",
  "gender": "M",
  "age": "26-35",
  "occupation": 10,
  "city_category": "A",
  "stay_in_current_city_years": "2",
  "marital_status": 0,
  "product_category_1": 3,
  "product_category_2": null,
  "product_category_3": null,
  "purchase": 8370,
  "timestamp": "2024-11-25T10:30:45.123Z",
  "event_type": "purchase"
}
```

### 3.3. Symulacja Strumienia Danych

#### Strategia Symulacji
1. **Batch Loading:** Wczytanie CSV do DataFrame
2. **Timestamp Generation:** Dodanie realistycznych timestampów
3. **Rate Control:** Kontrolowana emisja eventów (np. 100 events/sec)
4. **Randomization:** Losowa kolejność dla realizmu

#### Spark Streaming Consumer
```python
spark = SparkSession.builder \
    .appName("BlackFridayStreaming") \
    .getOrCreate()

# Read stream from Kafka
df_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "black-friday-purchases") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON and process
purchases_df = df_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")
```

### 3.4. Real-time Processing Pipeline

#### Window Operations
- **Tumbling Windows:** 5-minutowe okna dla agregacji
- **Sliding Windows:** 10-minutowe okna z 2-min przesunięciem
- **Session Windows:** Wykrywanie sesji zakupowych klienta

#### Agregacje w czasie rzeczywistym
1. Total sales per minute
2. Average purchase value per city
3. Top 10 products in rolling window
4. Customer purchase frequency

#### Przykładowa Agregacja
```python
# Real-time sales aggregation
sales_per_minute = purchases_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(window("timestamp", "1 minute")) \
    .agg(
        count("*").alias("transaction_count"),
        sum("purchase").alias("total_sales"),
        avg("purchase").alias("avg_purchase")
    )
```

### 3.5. Output Sinks

#### Streaming Outputs
1. **Console Sink:** Monitoring w czasie rzeczywistym
2. **Memory Sink:** Tymczasowe przechowywanie dla testów
3. **Delta Lake Sink:** Persystencja z ACID guarantees
4. **Parquet Sink:** Archiwizacja danych historycznych

```python
# Write to Delta Lake
query = sales_per_minute.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start("/delta/black-friday-sales")
```

---

## 4. PROCES UCZENIA MASZYNOWEGO NA KLASTRZE APACHE SPARK

### 4.1. Architektura ML Pipeline w Spark MLlib

#### Distributed Machine Learning
**Spark MLlib advantages:**
- **Scalability:** Training na datasets większych niż RAM
- **Distributed computing:** Parallel training across cluster nodes
- **Integration:** Seamless z Spark SQL i DataFrame API
- **Production-ready:** Spark ML Pipelines dla deployment

**Architecture:**
```
[Data] → [Feature Engineering] → [ML Pipeline] → [Model Training] → [Evaluation] → [Deployment]
   ↓            ↓                      ↓                ↓                 ↓              ↓
Spark DF   Transformers          Estimators      Cross-Val        Metrics        Serialization
```

### 4.2. Model 1: Regression - Predykcja Wartości Zakupu

#### Problem Statement
**Zadanie:** Przewidywanie wartości zakupu (`Purchase`) na podstawie cech klienta i produktu.
**Typ:** Supervised Learning - Regression
**Target variable:** Purchase (continuous, 185-23961 USD)

#### Feature Engineering dla Regression
```python
# Features selection
feature_cols = [
    "Gender_index", "Age_index", "Occupation",
    "City_Category_index", "Stay_In_Current_City_Years_index",
    "Marital_Status", "Product_Category_1",
    # Engineered features
    "user_total_purchases", "user_avg_purchase",
    "product_popularity", "category_avg_price"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
```

#### Model Training - Multiple Algorithms

**1. Linear Regression (baseline)**
```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="scaled_features",
    labelCol="Purchase",
    maxIter=100,
    regParam=0.1,  # L2 regularization
    elasticNetParam=0.5  # Mix of L1 and L2
)

lr_model = lr.fit(train_df)
```

**2. Random Forest Regressor**
```python
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="scaled_features",
    labelCol="Purchase",
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=10,
    seed=42
)

rf_model = rf.fit(train_df)
```

**3. Gradient Boosted Trees (GBT)**
```python
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(
    featuresCol="scaled_features",
    labelCol="Purchase",
    maxIter=100,
    maxDepth=5,
    stepSize=0.1
)

gbt_model = gbt.fit(train_df)
```

#### Hyperparameter Tuning
```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [5, 10, 20]) \
    .build()

# Cross-validation
evaluator = RegressionEvaluator(labelCol="Purchase", metricName="rmse")

cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,
    parallelism=4
)

cv_model = cv.fit(train_df)
best_model = cv_model.bestModel
```

#### Model Evaluation
```python
# Predictions
predictions = best_model.transform(test_df)

# Metrics
from pyspark.ml.evaluation import RegressionEvaluator

metrics = {
    "RMSE": RegressionEvaluator(metricName="rmse").evaluate(predictions),
    "MAE": RegressionEvaluator(metricName="mae").evaluate(predictions),
    "R2": RegressionEvaluator(metricName="r2").evaluate(predictions),
    "MSE": RegressionEvaluator(metricName="mse").evaluate(predictions)
}

# Feature importance (for tree-based models)
feature_importance = best_model.featureImportances
```

### 4.3. Model 2: Clustering - Segmentacja Klientów

#### Problem Statement
**Zadanie:** Grupowanie klientów w segmenty o podobnych zachowaniach zakupowych
**Typ:** Unsupervised Learning - Clustering
**Algorytm:** K-Means

#### Feature Engineering dla Clustering
```python
# RFM features per customer
customer_features = df.groupBy("User_ID").agg(
    # Recency (assumed from transaction order)
    count("*").alias("Frequency"),
    sum("Purchase").alias("Monetary"),
    avg("Purchase").alias("AvgPurchase"),
    stddev("Purchase").alias("StdPurchase"),
    countDistinct("Product_ID").alias("UniqueProducts"),
    countDistinct("Product_Category_1").alias("UniqueCategories")
)

# Normalize features
assembler = VectorAssembler(
    inputCols=["Frequency", "Monetary", "AvgPurchase", "UniqueProducts"],
    outputCol="features"
)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
```

#### Determining Optimal K
```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Elbow method
costs = []
silhouettes = []

for k in range(2, 11):
    kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features")
    model = kmeans.fit(customer_features)

    # Inertia (WSSSE)
    costs.append(model.summary.trainingCost)

    # Silhouette score
    predictions = model.transform(customer_features)
    evaluator = ClusteringEvaluator()
    silhouettes.append(evaluator.evaluate(predictions))

# Select optimal k based on elbow + silhouette
```

#### K-Means Training
```python
# Train with optimal k
kmeans = KMeans(
    k=5,  # Based on elbow analysis
    featuresCol="scaled_features",
    predictionCol="cluster",
    maxIter=100,
    seed=42
)

kmeans_model = kmeans.fit(customer_features)

# Cluster assignments
clustered_customers = kmeans_model.transform(customer_features)

# Cluster centers
centers = kmeans_model.clusterCenters()
```

#### Cluster Profiling
```python
# Analyze cluster characteristics
cluster_profiles = clustered_customers.groupBy("cluster").agg(
    count("*").alias("customer_count"),
    avg("Frequency").alias("avg_frequency"),
    avg("Monetary").alias("avg_monetary"),
    avg("AvgPurchase").alias("avg_ticket_size"),
    avg("UniqueProducts").alias("avg_unique_products")
)

# Assign business names to clusters
# Example: Cluster 0 = "High Value", Cluster 1 = "Occasional", etc.
```

### 4.4. Model 3: Recommendation System - Collaborative Filtering

#### Problem Statement
**Zadanie:** Rekomendacja produktów dla użytkowników na podstawie collaborative filtering
**Typ:** Recommendation System
**Algorytm:** ALS (Alternating Least Squares)

#### Data Preparation
```python
# Create user-product interaction matrix
# Rating = normalized purchase amount (implicit feedback)

interactions = df.groupBy("User_ID", "Product_ID").agg(
    sum("Purchase").alias("total_purchase"),
    count("*").alias("purchase_count")
)

# Normalize to 1-5 rating scale
from pyspark.sql.functions import percent_rank
from pyspark.sql.window import Window

window = Window.orderBy("total_purchase")
interactions = interactions.withColumn(
    "rating",
    (percent_rank().over(window) * 4 + 1)  # Scale to 1-5
)

# Index users and products
user_indexer = StringIndexer(inputCol="User_ID", outputCol="user_index")
product_indexer = StringIndexer(inputCol="Product_ID", outputCol="product_index")

interactions = user_indexer.fit(interactions).transform(interactions)
interactions = product_indexer.fit(interactions).transform(interactions)
```

#### ALS Model Training
```python
from pyspark.ml.recommendation import ALS

# Split data
train, test = interactions.randomSplit([0.8, 0.2], seed=42)

# ALS configuration
als = ALS(
    userCol="user_index",
    itemCol="product_index",
    ratingCol="rating",
    rank=10,  # Latent factors
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",  # Handle unseen users/items
    nonnegative=True,
    implicitPrefs=False
)

als_model = als.fit(train)
```

#### Evaluation
```python
# Predictions on test set
predictions = als_model.transform(test)

# RMSE
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)

# Precision@K, Recall@K
# Generate top 10 recommendations per user
user_recs = als_model.recommendForAllUsers(10)

# Generate top 10 users per product
product_recs = als_model.recommendForAllItems(10)
```

#### Generating Recommendations
```python
# Get recommendations for specific user
def get_recommendations(user_id, k=10):
    user_idx = user_indexer_model.transform(
        spark.createDataFrame([(user_id,)], ["User_ID"])
    ).select("user_index").first()[0]

    recs = als_model.recommendForUserSubset(
        spark.createDataFrame([(user_idx,)], ["user_index"]),
        k
    )

    # Convert back to product IDs
    return recs

# Example
recommendations = get_recommendations(1000001, k=10)
```

### 4.5. Model Deployment i Persistence

#### Saving Models
```python
# Save ML Pipeline
pipeline_model.write().overwrite().save("/models/purchase_prediction_pipeline")

# Save individual models
lr_model.write().overwrite().save("/models/linear_regression")
rf_model.write().overwrite().save("/models/random_forest")
kmeans_model.write().overwrite().save("/models/kmeans_segmentation")
als_model.write().overwrite().save("/models/als_recommendations")
```

#### Loading Models
```python
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel

# Load pipeline
loaded_pipeline = PipelineModel.load("/models/purchase_prediction_pipeline")

# Make predictions
predictions = loaded_pipeline.transform(new_data)
```

#### Real-time Scoring (Streaming)
```python
# Apply model to streaming data
scored_stream = loaded_pipeline.transform(streaming_df)

# Write predictions to output
scored_stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .start("/delta/predictions")
```

### 4.6. Experiment Tracking i MLOps

#### MLflow Integration (optional)
```python
import mlflow
import mlflow.spark

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 10)

    # Train model
    model = rf.fit(train_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.spark.log_model(model, "random_forest_model")
```

---


## 5. OCZEKIWANE REZULTATY

### Deliverables
1. **Jupyter Notebook** z pełną analizą EDA
2. **ETL Pipeline** w PySpark
3. **ML Models** (regression, classification, clustering, recommendation)
4. **Streaming Demo** z Kafka + Spark Streaming
5. **Dashboard** z wizualizacjami (Plotly/Dash)
6. **Dokumentacja Techniczna** (niniejszy dokument + więcej)
7. **Prezentacja** wyników i wniosków

### Metryki Sukcesu
- RMSE < 3000 dla prediction modelu
- R² > 0.85 dla regression
- Silhouette Score > 0.5 dla clustering
- Processing throughput > 1000 events/sec w streaming
- End-to-end latency < 5 sekund

---

## 6. WNIOSKI BIZNESOWE (przewidywane)

### Dla Retailerów
1. **Targeted Marketing:** Precyzyjna segmentacja klientów
2. **Inventory Optimization:** Przewidywanie popytu
3. **Pricing Strategy:** Optymalizacja cen na podstawie segmentów
4. **Personalization:** Rekomendacje produktów real-time

### Dla Decision Makers
1. **Revenue Forecasting:** Lepsze planowanie finansowe
2. **Customer Lifetime Value:** Identyfikacja high-value customers
3. **Campaign Effectiveness:** ROI z kampanii marketingowych
4. **Churn Prevention:** Early warning system



