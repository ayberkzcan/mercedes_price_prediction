# Mercedes Price Analysis

This project analyzes used Mercedes car prices using a dataset containing features such as year, mileage, fuel type, and transmission. The goal is to explore the data, identify key trends, and provide insights for data-driven pricing decisions.

---

## 📊 Exploratory Data Analysis & Insights

## Price Distribution
![Price Distribution](images/price_distribution.png)  
- Most prices are concentrated in the 20k–40k range, representing mid-segment vehicles.  
- Outliers in the top 1% were removed to create a more symmetric distribution.  
- **Business Insight:** Pricing for mid-segment vehicles can be standardized, while extreme luxury models may require special pricing strategies.

## Mileage vs Price
![Mileage vs Price](images/mileage_price_after.png)  
- As mileage increases, price decreases (negative correlation).  
- High-mileage cars consistently sell for less.  
- **Business Insight:** Vehicle age and mileage are the strongest predictors of price. Sellers can optimize pricing based on these variables.

## Vehicle Count by Year
![Vehicle Count by Year](images/year_count.png)  
- Newer vehicles dominate the dataset; older cars are underrepresented.  
- **Business Insight:** Inventory is skewed toward newer models, so pricing and marketing strategies should focus on recent vehicles. Model predictions for older cars may be less reliable.

## Correlation Heatmap
![Correlation Heatmap](images/correlation.png)  
- Mileage and year have the highest impact on price.  
- Other features have weaker correlations but may still provide incremental insights.  
- **Business Insight:** Key features can guide promotions, discounts, and valuation strategies.

---

## 📈 Model Overview (Optional)
A regression model was trained to understand pricing relationships.  

| Metric | Value |
|--------|-------|
| MAE    | 2392.63 |
| RMSE   | 3468.27 |
| R²     | 0.879 |

> The model captures general price trends, particularly for mid-segment vehicles.

---

## ⚡ Key Takeaways

- Vehicle age and mileage are the most influential features affecting price.  
- Outlier removal improves data consistency and visualization clarity.  
- Sellers and businesses can use these insights to optimize pricing and inventory decisions.  
- This project demonstrates **EDA + business insights**, making it suitable for a data analyst portfolio.

---

## 🐍 Python Version

- Python 3.10

---

## ▶️ How to Run (Optional)

1. Place `merc.csv` in the same folder as the notebook/script  
2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
