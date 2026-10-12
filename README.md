# Unsupervised Anomaly Detection on 1.07M NYC Property Records

Building an explainable investigation queue when **no fraud labels exist**.
Academic project · USC Marshall, DSO 562 Fraud Analytics · Fall 2024 · individual submission

**Result:** ranked all 1,044,492 in-scope properties by anomaly score and produced a top-10,000 review queue, with variable-level z-scores showing why each record scored high.

---

## Why this problem is different

Without confirmed fraud labels you cannot train a classifier or report precision/recall.
The task is to surface records whose value-to-size relationships look abnormal **compared with their peers**, then hand investigators a ranked, explainable list.

## Data

- NYC property valuation and assessment records (NYC Open Data), 1,070,994 rows × 32 fields
- Excluded government, public, and cemetery owners plus easements (26,502 records) → 1,044,492 modeled
- Raw data is not included in this repo (≈180 MB public file)

## Approach

**1. Data quality and imputation**
- ZIP (20,431 missing): restored by same street address + borough (2,832), then neighboring records with matching ZIPs (16,126), then forward-fill (1,473)
- Values (FULLVAL, AVLAND, AVTOT): zeros treated as missing and filled with group means, from tax class + borough + building class down to tax class
- Stories and lot/building dimensions: group mode/mean with invalid 0/1 values treated as missing

**2. Feature engineering: 29 anomaly signals**
- 9 base ratios: three value fields ÷ lot size, building footprint, and building volume
- 18 peer-normalized ratios: each base ratio ÷ the ZIP average and ÷ the tax-class average
- 2 extra ratios: value ratio and size ratio
- Each ratio is median-normalized and the larger of the value and its inverse is kept, so **abnormally high and abnormally low** values both score

**3. Dimensionality reduction**
z-scale → PCA → 5 components (75.6% cumulative variance) → z-scale again so each component weighs equally.

**4. Two anomaly scores, rank-combined**
- Distance from the center in PCA space (Minkowski, p = 2)
- Autoencoder (MLP, 3 hidden layers) reconstruction error
- Each score is converted to a rank and the two ranks are averaged 50:50, so neither score's scale dominates

**5. Case review**
The top record ranked #1 on both scores. Reviewing high-ranked cases showed that some anomalies have legitimate explanations, such as a medical center that slipped through the owner-keyword exclusion or HOA common-area parcels.
That points to where exclusion rules need refinement.

## Limitations

- No labels, so **no precision, recall, or fraud prevented is claimed**. The output is a review priority, not a fraud probability.
- The final ZIP forward-fill assumes geographic ordering; production use would need authoritative geocoding.
- In production I would measure precision@K from investigator dispositions and recalibrate the queue.

## Repository

```
notebooks/
  01_data_quality_report.ipynb       # 32-field data quality review
  02_unsupervised_anomaly_scoring.ipynb  # imputation, features, PCA, scores, ranking
report/final_report.pdf
```

Notebooks build on course-provided starter code; exclusion logic, imputation, feature design, scoring, and case review are my own work.

**Stack:** Python · pandas · NumPy · scikit-learn (PCA, MLP autoencoder) · Matplotlib / Seaborn
