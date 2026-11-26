# DataCo Quality Audit (measured on the raw file as downloaded)

- Source file: data/raw/DataCoSupplyChainDataset.csv (95.9 MB)
- Rows: 180,519 | Columns: 53
- Exact duplicate rows: 0

- Order dates parseable: 180,519 (100.0%); range 2015-01-01 00:00:00 to 2018-01-31 23:38:00

- Target balance (Late_delivery_risk): 0: 45.2%, 1: 54.8%

## Missing values (columns with any)

- Product Description: 100.0%
- Order Zipcode: 86.2%

## PII / credential columns present in raw file (dropped at load, never features)

- Customer Email
- Customer Password
- Customer Fname
- Customer Lname
- Customer Street
- Customer Zipcode
- Product Image
- Order Customer Id
- Customer Id

## Leakage columns (banned from features)

- Days for shipping (real)
- Delivery Status
- Late_delivery_risk
- shipping date (DateOrders)

## Other post-outcome columns excluded (docs/02)

- Days for shipping (real): the realized outcome itself - direct leakage
- Delivery Status: derived from realized vs scheduled dates - direct leakage
- shipping date (DateOrders): known only once shipped - post-outcome
- Benefit per order: profit computed after fulfilment - post-outcome
- Order Profit Per Order: profit computed after fulfilment - post-outcome
- Order Item Profit Ratio: derived from post-fulfilment profit - post-outcome
- Sales per customer: aggregate computed over the full dataset window - temporal leakage risk

## Anomalies

- Scheduled days: min 0, max 4, negative values: 0
- Real shipping days: min 0, max 6, negative values: 0
- Rows where real < 0 or scheduled <= 0: 9737
- Negative Sales rows: 0
