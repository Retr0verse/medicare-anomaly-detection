# medicare-anomaly-detection

Analyzing Medicare provider payment anomalies using CMS data, statistical methods, and exploratory analysis.



\# Medicare Anomaly Detection



Analyzing Medicare provider payment anomalies using CMS data, statistical methods, and exploratory analysis.



\## Team

\- Jonathan Kennedy

\- Edwin Sosi

\- Deborah Cuellar

\- Laura Casillas



\## Project Goal

Identify anomalous provider behavior using Medicare provider-service data. This project focuses on statistical outliers, not proving fraud.



\## Dataset

CMS Medicare Physician \& Other Practitioners (2023)



\## Current Progress

\- Loaded CMS dataset (\~1.1M providers)

\- Performed Z-score analysis on provider-level payments

\- Identified \~6,200 providers above 3 standard deviations



\## Initial Finding

Provider payment distribution is highly skewed, with a subset of extreme outliers identified using Z-score analysis.



\## Project Structure

\- src/ → analysis scripts

\- notebooks/ → exploration

\- reports/ → deliverables

\- data/ → local data only (not tracked)

