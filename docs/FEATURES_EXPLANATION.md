# Feature Engineering - Szczegółowe Wyjaśnienie Cech

## Spis Treści
1. [User-Level Features](#1-user-level-features)
2. [Product-Level Features](#2-product-level-features)
3. [Category-Level Features](#3-category-level-features)
4. [RFM Features](#4-rfm-features)
5. [Interaction Features](#5-interaction-features)
6. [Jak używać tych cech](#jak-używać-tych-cech)

---

## 1. User-Level Features (Cechy na poziomie użytkownika)

### Co to jest?
Agregacje statystyk zakupowych dla każdego użytkownika - pokazują **historyczne zachowania zakupowe**.

### Cechy:

#### `user_purchase_count` (Liczba zakupów)
**Co to:** Ile razy użytkownik coś kupił
```python
User_ID: 1000001 → user_purchase_count: 45
# Ten użytkownik zrobił 45 zakupów
```

**Po co:**
- Lojalność klienta - czy kupuje często czy rzadko?
- High-frequency buyers vs one-time shoppers
- Marketing: Często kupujący → loyalty rewards
- Rarely kupujący → reactivation campaigns

**Jak używać:**
- W modelach ML do predykcji: "Jak często klient kupuje?"
- Segmentacja: "Casual buyers" vs "Power users"
- Business: Kogo warto targować kampanią?

---

#### `user_total_spent` (Całkowita kwota wydana)
**Co to:** Suma wszystkich zakupów użytkownika
```python
User_ID: 1000001 → user_total_spent: $50,000
# Ten użytkownik wydał $50k w sumie
```

**Po co:**
- Customer Lifetime Value (CLV) - jak wartościowy jest klient?
- VIP customers identification
- Revenue forecasting per customer

**Jak używać:**
- Segmentacja: High-value vs Low-value customers
- Personalizacja: VIP → premium products, special offers
- Model ML: Feature koreluje z przyszłymi zakupami

---

#### `user_avg_purchase` (Średnia wartość zakupu)
**Co to:** Średni "koszyk" użytkownika
```python
user_total_spent: $50,000 / user_purchase_count: 45 = user_avg_purchase: $1,111
```

**Po co:**
- Average Order Value (AOV) - ile wydaje per transakcja
- Price sensitivity - czy kupuje drogie czy tanie rzeczy?

**Jak używać:**
- Pricing strategy: Użytkownicy z high AOV → premium pricing OK
- Cross-sell opportunities: Low AOV → bundle deals
- Model: Feature do predykcji przyszłej wartości zakupu

---

#### `user_std_purchase` (Odchylenie standardowe zakupów)
**Co to:** Jak różnorodne są zakupy użytkownika
```python
Użytkownik A: zakupy [100, 110, 105, 95] → std = 6.45 (consistent)
Użytkownik B: zakupy [50, 500, 100, 1000] → std = 435.89 (varied)
```

**Po co:**
- Predictability klienta
- Consistent buyers → stabilny revenue
- Varied buyers → unpredictable, może kupić bardzo dużo lub mało

**Jak używać:**
- Risk assessment: High std → high variance, hard to predict
- Kampanie: Consistent buyers → regular promotions work
- Model: Pomaga w uncertainty estimation

---

#### `user_unique_products` (Liczba unikalnych produktów)
**Co to:** Ile różnych produktów użytkownik kupił
```python
User kupił: [ProductA, ProductA, ProductB, ProductC, ProductA]
user_unique_products = 3
```

**Po co:**
- Product diversity - czy kupuje tylko jedno czy eksploruje?
- Engagement level
- Cross-category shoppers vs single-category loyalists

**Jak używać:**
- Recommendation system: Low diversity → recommend new categories
- Segmentacja: "Explorers" vs "Specialists"
- Upsell: Single-product users → introduce similar products

---

#### `user_unique_categories` (Liczba unikalnych kategorii)
**Co to:** Ile różnych kategorii produktów użytkownik eksploruje
```python
User kupował z kategorii: [Electronics, Clothing, Electronics, Food, Electronics]
user_unique_categories = 3
```

**Po co:**
- Category affinity
- Multi-category shoppers → higher CLV
- Specialization pattern

**Jak używać:**
- Cross-category campaigns
- Bundle offers across categories
- Model: Feature do recommendation systemów

---

## 2. Product-Level Features (Cechy produktu)

### Co to jest?
Statystyki konkretnego produktu - jego popularność, cena, jak często kupowany.

### Cechy:

#### `product_purchase_count` (Popularność produktu)
**Co to:** Ile razy produkt został kupiony (total)
```python
Product_ID: P00069042 → product_purchase_count: 1,523
# Ten produkt kupiono 1,523 razy
```

**Po co:**
- Bestsellers identification
- Inventory management - popularne → więcej stock
- Social proof - "1,523 people bought this"

**Jak używać:**
- Homepage: Feature bestsellers
- Recommendations: "Popular items you might like"
- Inventory: High count → never run out
- Model: Popular products → easier to sell

---

#### `product_unique_users` (Reach produktu)
**Co to:** Ile różnych użytkowników kupiło ten produkt
```python
Product kupiony przez 500 unikalnych userów (niektórzy kupili 2x)
product_unique_users = 500
```

**Po co:**
- Market penetration
- Viral potential
- Customer base size

**Jak używać:**
- Marketing: High reach → mass appeal product
- Targeting: Low reach → niche product, targeted ads
- Model: Wide appeal → easier to recommend

---

#### `product_avg_price` (Średnia cena produktu)
**Co to:** Średnia cena po jakiej sprzedawany był produkt
```python
Product sprzedany po: [$100, $95, $110, $105]
product_avg_price = $102.5
```

**Po co:**
- Price point identification
- Discount analysis - czy było promotion?
- Market positioning - premium vs budget

**Jak używać:**
- Pricing strategy
- Discount planning
- Model: Price feature do prediction

---

#### `product_popularity_score` (Normalized popularity 0-1)
**Co to:** Ranking popularności znormalizowany
```python
Najlepszy produkt: 1.0
Najgorszy produkt: 0.0
Średni produkt: ~0.5
```

**Po co:**
- Fair comparison między produktami
- Easier to use in ML models (same scale)

**Jak używać:**
- Ranking: Top 10% (score > 0.9)
- Recommendations: Recommend products with score > 0.7
- Model: Normalized feature → better training

---

#### `product_popularity_rank` (Ranking)
**Co to:** Pozycja w rankingu (1 = najbardziej popularny)
```python
Rank 1: Best seller
Rank 100: Less popular
Rank 1000: Niche product
```

**Po co:**
- Clear hierarchy
- Tier system (Tier 1: ranks 1-100, etc.)

**Jak używać:**
- Merchandising: Top 100 → featured section
- Inventory: Top 50 → always in stock
- Marketing: "Rank #5 bestseller!"

---

## 3. Category-Level Features (Cechy kategorii)

### Co to jest?
Statystyki całej kategorii produktów.

#### `category_avg_price` (Średnia cena w kategorii)
**Co to:** Średnia cena wszystkich produktów w tej kategorii
```python
Category "Electronics": category_avg_price = $500
Category "Clothing": category_avg_price = $50
```

**Po co:**
- Price benchmarking
- Category positioning (premium vs budget)
- Understanding category economics

**Jak używać:**
- Pricing: Produkt w Electronics po $100 → cheap dla kategorii
- Model: Comparison feature (product_price / category_avg_price)
- Business: Which categories are high-margin?

---

#### `category_total_revenue` (Całkowity przychód kategorii)
**Co to:** Suma sprzedaży w kategorii
```python
Electronics: $10M
Clothing: $2M
Food: $500K
```

**Po co:**
- Revenue distribution
- Strategic focus - które kategorie są najważniejsze?

**Jak używać:**
- Business strategy: Focus on high-revenue categories
- Marketing budget allocation
- Inventory investment

---

## 4. RFM Features (Recency, Frequency, Monetary)

### Co to jest?
Framework marketingowy do oceny wartości klienta w 3 wymiarach.

#### `rfm_frequency_score` (0-1)
**Co to:** Jak często kupuje (znormalizowane)
```python
0.0 = Kupuje najrzadziej (1-2 zakupy)
0.5 = Average
1.0 = Kupuje najczęściej (power user)
```

**Po co:**
- Engagement level
- Loyalty indicator

**Jak używać:**
- Score < 0.3 → Casual buyer, needs reactivation
- Score > 0.7 → Loyal customer, retention programs

---

#### `rfm_monetary_score` (0-1)
**Co to:** Jak dużo wydaje (znormalizowane)
```python
0.0 = Lowest spender
0.5 = Average spender
1.0 = Highest spender (whale)
```

**Po co:**
- Revenue contribution
- Value segmentation

**Jak używać:**
- Score > 0.8 → VIP treatment, premium access
- Score < 0.2 → Budget-friendly offers

---

#### `rfm_score` (Combined 0-1)
**Co to:** Średnia frequency + monetary
```python
rfm_score = (rfm_frequency_score + rfm_monetary_score) / 2
```

**Po co:**
- Single metric dla customer value
- Prostsza segmentacja

**Jak używać:**
```python
rfm_score > 0.8 → "Champions" - best customers
rfm_score 0.5-0.8 → "Loyal Customers"
rfm_score 0.3-0.5 → "Potential"
rfm_score < 0.3 → "At Risk"
```

**Segmentacja:**
- Champions (>0.8): VIP programs, early access
- Loyal (0.5-0.8): Retention, cross-sell
- Potential (0.3-0.5): Upsell campaigns
- At Risk (<0.3): Reactivation, discounts

---

## 5. Interaction Features (Cechy interakcyjne)

### Co to jest?
Porównania i relacje między różnymi cechami - pokazują **anomalie i wzorce**.

#### `purchase_vs_user_avg_ratio`
**Co to:** Czy ten zakup jest większy/mniejszy od średniej użytkownika
```python
Purchase: $500
user_avg_purchase: $100
ratio = $500 / $100 = 5.0

ratio > 1.0 → Kupił więcej niż zwykle (unusual high)
ratio < 1.0 → Kupił mniej niż zwykle
ratio ≈ 1.0 → Typowy zakup
```

**Po co:**
- Anomaly detection
- Special occasions (gifts?)
- Upsell success indicator

**Jak używać:**
- Ratio > 2.0 → "Big purchase! Want extended warranty?"
- Ratio < 0.5 → "Budget shopping today?"
- Model: Catches unusual behavior

---

#### `purchase_vs_product_avg_ratio`
**Co to:** Czy użytkownik przepłacił/dostał discount
```python
User zapłacił: $80
Product avg price: $100
ratio = 0.8 → Dostał 20% discount

ratio > 1.0 → Zapłacił więcej (premium variant?)
ratio < 1.0 → Dostał discount/promotion
```

**Po co:**
- Price sensitivity analysis
- Promotion effectiveness
- Discount hunting behavior

**Jak używać:**
- Ratio < 0.8 → Discount shopper, target with sales
- Ratio > 1.0 → Premium buyer, upsell works

---

#### `is_above_user_avg` (Binary flag 0/1)
**Co to:** Prosty flag - czy ten zakup > średnia użytkownika
```python
1 = TAK, większy zakup
0 = NIE, mniejszy/równy
```

**Po co:**
- Easier to interpret
- Good for classification models

**Jak używać:**
- Filter: Show only above-average purchases
- Model: Binary feature for decision trees

---

#### `is_high_value_customer` (Binary flag 0/1)
**Co to:** Czy użytkownik jest w top 20% (rfm_score >= 0.8)
```python
1 = TAK, VIP customer
0 = NIE, regular customer
```

**Po co:**
- Instant VIP identification
- Priority handling

**Jak używać:**
- CRM: Auto-flag VIP customers
- Customer service: Priority queue
- Marketing: Exclusive offers
- Model: Important segmentation feature

---

## Jak Używać Tych Cech

### 1. W Modelach Machine Learning

#### Regression (Predykcja Purchase)
```python
Important features:
- user_avg_purchase ← Historia użytkownika
- product_avg_price ← Cena produktu
- category_avg_price ← Kontekst kategorii
- rfm_monetary_score ← Spending power
- purchase_vs_user_avg_ratio ← Anomalie
```

**Dlaczego:**
- user_avg_purchase: Najlepszy predictor przyszłego zachowania
- product_avg_price: Bazowa cena
- Ratios: Catches special cases

---

#### Clustering (Segmentacja klientów)
```python
Features to use:
- user_purchase_count ← Frequency
- user_total_spent ← Monetary
- user_avg_purchase ← AOV
- user_unique_categories ← Diversity
- rfm_score ← Overall value
```

**Segmenty powstają:**
```
Cluster 0: "Champions"
  - High purchase count
  - High total spent
  - High rfm_score

Cluster 1: "Bargain Hunters"
  - Medium purchase count
  - Low avg purchase
  - Many unique products

Cluster 2: "Occasional Big Spenders"
  - Low purchase count
  - High avg purchase
  - High std_purchase
```

---

#### Recommendation System (ALS)
```python
Auxiliary features:
- product_popularity_score ← Boost popular items
- user_unique_categories ← Cross-category recs
- category_avg_price ← Price matching
```

**Jak:**
- Popular products → Cold start problem
- user_unique_categories: Jeśli = 1, recommend same category
- If > 3, recommend cross-category

---

### 2. W Business Analytics

#### Dashboard KPIs
```python
Key Metrics:
1. Average RFM Score: 0.45 (↑ 5% MoM)
2. % High Value Customers: 18% (top 20%)
3. Average AOV: $1,234
4. Top Products by popularity_score
```

---

#### Customer Segmentation Report
```sql
SELECT
  CASE
    WHEN rfm_score >= 0.8 THEN 'Champions'
    WHEN rfm_score >= 0.5 THEN 'Loyal'
    WHEN rfm_score >= 0.3 THEN 'Potential'
    ELSE 'At Risk'
  END as segment,
  COUNT(*) as customer_count,
  AVG(user_total_spent) as avg_clv
FROM user_features
GROUP BY segment
```

---

#### Marketing Campaigns
```python
# Campaign 1: Reactivate low-frequency users
target = users.filter(
    (col("rfm_frequency_score") < 0.3) &
    (col("user_total_spent") > 1000)  # Had money before
)
# Action: "We miss you! Here's 20% off"

# Campaign 2: Upsell to high-value customers
target = users.filter(
    (col("is_high_value_customer") == 1) &
    (col("user_unique_categories") <= 2)  # Don't shop across categories
)
# Action: "Explore our [other category]"

# Campaign 3: Cross-sell
target = users.filter(
    col("user_avg_purchase") > 500  # Can afford
).join(
    products.filter(col("product_popularity_score") > 0.7)  # Popular items
)
# Action: "You might also like..."
```

---

### 3. Real-time Decision Making

#### At Checkout:
```python
if purchase_vs_user_avg_ratio > 2.0:
    show("Big purchase! Add insurance for $10?")

if is_high_value_customer == 1:
    show("Free express shipping for you!")

if user_unique_products < 3:
    show("First time buying [category]? Here's a guide")
```

---

#### Personalization Engine:
```python
def get_homepage_layout(user_id):
    if rfm_score > 0.8:
        return "premium_layout"  # Show high-end products
    elif rfm_frequency_score < 0.3:
        return "discount_layout"  # Emphasize sales
    elif user_unique_categories > 5:
        return "explorer_layout"  # Show variety
    else:
        return "standard_layout"
```

---

## Podsumowanie: Dlaczego Te Cechy Są Ważne

### 1. **Kontekst Historyczny**
Cechy User-level dają context: "Jak ten user zachowywał się w przeszłości?"
→ Najlepszy predictor przyszłości to przeszłość

### 2. **Relative Information**
Interaction features (ratios): "Czy to normalne czy anomalia?"
→ Łapią special cases, które raw features pominęłyby

### 3. **Market Intelligence**
Product/Category features: "Jak popularne jest to na rynku?"
→ Social proof, market positioning

### 4. **Customer Value**
RFM features: "Jak wartościowy jest ten klient?"
→ Business prioritization, resource allocation

### 5. **Scalability dla ML**
Wszystkie features są znormalizowane (0-1) lub już na dobrej skali
→ Models train faster and better

---

## Quick Reference: Kiedy Użyć Której Cechy

| Use Case | Kluczowe Features |
|----------|-------------------|
| **Predicting next purchase amount** | user_avg_purchase, product_avg_price, rfm_monetary_score |
| **Customer segmentation** | rfm_score, user_purchase_count, user_total_spent |
| **Churn prediction** | rfm_frequency_score, user_purchase_count (last 30d) |
| **Product recommendations** | product_popularity_score, user_unique_categories |
| **Fraud detection** | purchase_vs_user_avg_ratio (high = suspicious) |
| **VIP identification** | is_high_value_customer, rfm_score |
| **Inventory planning** | product_purchase_count, product_popularity_rank |
| **Pricing strategy** | category_avg_price, product_avg_price |
| **Marketing targeting** | rfm_score, user_unique_categories |

---

**Autor:** Vlad Usatenko
**Data:** 2025-11-25
**Projekt:** Black Friday Big Data Analysis
